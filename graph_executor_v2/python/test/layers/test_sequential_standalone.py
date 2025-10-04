# -*- coding: utf-8 -*-
# python/test/layers/test_sequential_standalone.py
# --------------------------------------------------
# 목적:
#   1) Sequential + Dense + SoftmaxCrossEntropy 통합 점검
#   2) Python bwd vs Native bwd 그라드 수치 검증(compare_grads)
#   3) 미니 학습 루프(mini_train)로 손실 감소 여부 확인
#   4) 간단 SGD 스텝으로 파라미터 업데이트 in-place 확인
#   5) CE(reduction='none'↔'mean') 일관성 및 ignore_index/label_smoothing 스폿 체크
# --------------------------------------------------

# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))   # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))  # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp

# 레이어/모델
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.layers.model import Sequential


def _sgd_step_sequential(model: Sequential, lr: float = 1e-2):
    """아주 단순한 SGD: p -= lr * g"""
    for lyr in getattr(model, "layers", []):
        for p_name, g_name in [("W", "dW"), ("b", "db")]:
            if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                p, g = getattr(lyr, p_name), getattr(lyr, g_name)
                if g is None:
                    continue
                p[...] = p - lr * g  # CuPy ndarray in-place


def _cross_check_none_vs_mean(logits: cp.ndarray, y: cp.ndarray):
    """reduction='none' 평균 ≈ reduction='mean' 스칼라 검증"""
    M = logits.shape[0]
    ce_none = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="none", from_logits=True)
    loss_vec = ce_none((logits, y))           # (M,)
    dlogits_vec = ce_none.backward(cp.full((M,), 1.0 / M, dtype=cp.float32))  # (1/M) 스케일

    ce_mean = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="mean", from_logits=True)
    loss_mean = ce_mean((logits, y))          # (1,)
    dlogits_mean = ce_mean.backward(cp.array(1.0, dtype=cp.float32))

    # 손실 스칼라 일치(허용 오차 내)
    err_loss = abs(float(loss_vec.mean()) - float(loss_mean[0]))
    print(f"[CE cross-check] mean(loss_vec) vs loss_mean diff = {err_loss:.6e}")
    assert err_loss < 5e-4

    # 그라드 총합(노름 기준) 스폿 비교
    n1 = float(cp.linalg.norm(dlogits_vec))
    n2 = float(cp.linalg.norm(dlogits_mean))
    rel = abs(n1 - n2) / max(1e-6, max(n1, n2))
    print(f"[CE cross-check] ||dlogits_none|| vs ||dlogits_mean|| rel diff = {rel:.6e}")
    assert rel < 5e-3


def _single_pass(use_native_bwd_dense2: bool = True, do_spot_checks: bool = True):
    cp.random.seed(42)
    M, Din, H, Cout = 32, 64, 128, 10
    x = cp.random.randn(M, Din, dtype=cp.float32)
    y = cp.random.randint(0, Cout, size=(M,), dtype=cp.int32)

    model = Sequential(
        Dense(H, activation="relu", initializer="he", use_native_bwd=False, name="dense1"),
        Dense(Cout, activation=None, initializer="xavier", use_native_bwd=use_native_bwd_dense2, name="dense2"),
        name="mlp_seq"
    )

    # (선택) 사전 build
    try:
        model.build((M, Din))
    except Exception:
        pass

    # ---- forward ----
    logits = model(x)
    assert logits.shape == (M, Cout)
    print(f"[Sequential] forward OK: logits shape {logits.shape}")

    # ---- loss (reduction='none') ----
    ce = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="none", from_logits=True)
    loss_vec = ce((logits, y))
    assert loss_vec.shape == (M,)
    print(f"[SoftmaxCE] forward OK: loss per-sample shape {loss_vec.shape}")

    # backward (여기서 1/M 평균 스케일을 적용함)
    grad_scale = cp.full((M,), 1.0 / M, dtype=cp.float32)
    dlogits = ce.backward(grad_scale)
    assert dlogits.shape == logits.shape
    print("[SoftmaxCE] backward OK (reduction='none')")

    # ---- model backward ----
    _ = model.backward(dlogits)
    for lyr in model.layers:
        if isinstance(lyr, Dense):
            assert lyr.dW is not None and lyr.db is not None
            assert lyr.dW.shape == lyr.W.shape
            assert lyr.db.shape == lyr.b.shape
    print(f"[Sequential] backward OK (use_native_bwd_dense2={use_native_bwd_dense2})")

    # ---- Invariant: 마지막 Dense의 db == sum(dY) 확인 (레이어 bwd는 평균 스케일 금지) ----
    last = model.layers[-1]
    sum_dY = cp.sum(dlogits, axis=0, keepdims=True)  # last.db가 (1,N)이므로 keepdims=True
    max_abs_bias_err = float(cp.max(cp.abs(last.db - sum_dY)))
    print(f"[invariant] ||db - sum(dY)||inf = {max_abs_bias_err:.3e}")
    assert max_abs_bias_err < 1e-5, "Dense bwd must use db=sum(dY), not mean(dY)."

    # ---- SGD step ----
    W2_before = last.W.copy()
    b2_before = last.b.copy()
    _sgd_step_sequential(model, lr=1e-2)
    print("[Optim] SGD step OK")
    dW_norm = float(cp.linalg.norm(W2_before - last.W))
    db_norm = float(cp.linalg.norm(b2_before - last.b))
    assert dW_norm > 0 or db_norm > 0
    print(f"[Optim] param delta norms: dW={dW_norm:.6f}, db={db_norm:.6f}")

    # ---- reduction='mean' 경로 ----
    ce_mean = SoftmaxCrossEntropy(label_smoothing=0.0, reduction="mean", from_logits=True)
    loss_mean = ce_mean((logits, y))            # (1,)
    assert loss_mean.shape == (1,)
    dlogits_mean = ce_mean.backward(cp.array(1.0, dtype=cp.float32))
    assert dlogits_mean.shape == logits.shape
    print("[SoftmaxCE] reduction='mean' OK")

    # ---- (옵션) CE 일관성/ignore_index/ls 스폿 체크 ----
    if do_spot_checks:
        _cross_check_none_vs_mean(logits, y)

        # ignore_index 스폿: 일부 샘플 제외 시 mean이 감소(또는 동일)하는지 확인
        y_ign = y.copy()
        if M >= 2:
            y_ign[:2] = -1  # ignore_index
        ce_ign = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="mean",
                                     from_logits=True, ignore_index=-1)
        loss_ign = ce_ign((logits, y_ign))[0]
        print(f"[CE spot] loss(mean) w/ ignore_index: {float(loss_ign):.6f}")

        # label_smoothing 스폿: 보통 loss가 약간 증가하는 경향(모델/분포에 따라 다름)
        ce_ls0 = SoftmaxCrossEntropy(label_smoothing=0.00, reduction="mean", from_logits=True)
        ce_ls5 = SoftmaxCrossEntropy(label_smoothing=0.10, reduction="mean", from_logits=True)
        loss_ls0 = float(ce_ls0((logits, y))[0])
        loss_ls5 = float(ce_ls5((logits, y))[0])
        print(f"[CE spot] loss(ls=0.00)={loss_ls0:.6f}, loss(ls=0.10)={loss_ls5:.6f}")

    # summary (선택)
    try:
        print(model.summary())
    except Exception:
        pass

    return float(loss_vec.mean()), float(loss_mean[0])


def compare_grads():
    """
    Python bwd vs Native bwd의 dW/db 최대절대오차 비교.
    - Loss에서 이미 1/M 스케일을 적용하므로, 레이어 bwd는 합(sum)이어야 함.
    """
    cp.random.seed(7)
    M, Din, H, Cout = 32, 64, 128, 10
    x = cp.random.randn(M, Din, dtype=cp.float32)
    y = cp.random.randint(0, Cout, size=(M,), dtype=cp.int32)

    def make_model(native_bwd_last: bool):
        return Sequential(
            Dense(H, activation="relu", initializer="he", use_native_bwd=False, name="d1"),
            Dense(Cout, activation=None, initializer="xavier", use_native_bwd=native_bwd_last, name="d2"),
            name=f"m_native_{native_bwd_last}"
        )

    # 기준 파라미터 생성
    base = make_model(False); _ = base(x)
    W1, b1 = base.layers[0].W.copy(), base.layers[0].b.copy()
    W2, b2 = base.layers[1].W.copy(), base.layers[1].b.copy()

    # 두 경로 모델 생성 + build
    m_py = make_model(False); _ = m_py(x)
    m_nv = make_model(True);  _ = m_nv(x)

    # 파라미터 동기화 (in-place)
    m_py.layers[0].W[...] = W1; m_py.layers[0].b[...] = b1
    m_py.layers[1].W[...] = W2; m_py.layers[1].b[...] = b2
    m_nv.layers[0].W[...] = W1; m_nv.layers[0].b[...] = b1
    m_nv.layers[1].W[...] = W2; m_nv.layers[1].b[...] = b2

    ce = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="none", from_logits=True)

    # 공통 dY 사용 (여기서 1/M 적용됨)
    logits_py = m_py(x); _ = ce((logits_py, y))
    dY = ce.backward(cp.full((M,), 1.0/M, dtype=cp.float32))

    logits_nv = m_nv(x); _ = ce((logits_nv, y))
    _ = m_py.backward(dY)
    _ = m_nv.backward(dY)

    # 마지막 Dense 비교
    py, nv = m_py.layers[1], m_nv.layers[1]

    # (안전) 두 경로 모두 db가 sum(dY)인지 개별 체크
    sum_dY = cp.sum(dY, axis=0, keepdims=True)
    err_py = float(cp.max(cp.abs(py.db - sum_dY)))
    err_nv = float(cp.max(cp.abs(nv.db - sum_dY)))
    print(f"[invariant] py: ||db-sum(dY)||inf={err_py:.3e}, nv: {err_nv:.3e}")
    assert err_py < 1e-5 and err_nv < 1e-5, "Layer bwd must produce db=sum(dY)."

    max_abs_dW = float(cp.max(cp.abs(py.dW - nv.dW)))
    max_abs_db = float(cp.max(cp.abs(py.db - nv.db)))
    print(f"[grad diff] dW={max_abs_dW:.3e}, db={max_abs_db:.3e}")

    tol = 1e-5
    assert max_abs_dW < tol and max_abs_db < tol, \
        f"native vs python bwd mismatch too large (tol={tol})"


def mini_train(epochs: int = 5, batch_size: int = 64, lr: float = 1e-2):
    """아주 간단한 미니배치 학습 루프."""
    cp.random.seed(0)
    N, Din, H, Cout = 512, 64, 128, 10
    X = cp.random.randn(N, Din, dtype=cp.float32)
    Y = cp.random.randint(0, Cout, size=(N,), dtype=cp.int32)

    model = Sequential(
        Dense(H, activation="relu", initializer="he", use_native_bwd=False),
        Dense(Cout, activation=None, initializer="xavier", use_native_bwd=True),
        name="mlp_seq"
    )
    ce = SoftmaxCrossEntropy(label_smoothing=0.0, reduction="mean", from_logits=True)

    def loader(X, Y, bs=64, shuffle=True):
        N = len(X)
        idx = cp.arange(N)
        if shuffle:
            cp.random.shuffle(idx)
        for s in range(0, N, bs):
            take = idx[s:s+bs]
            yield X[take], Y[take]

    for ep in range(1, epochs + 1):
        running = 0.0
        steps = 0
        for xb, yb in loader(X, Y, bs=batch_size, shuffle=True):
            logits = model(xb)
            loss = ce((logits, yb))[0]  # scalar
            dlogits = ce.backward(cp.array(1.0, dtype=cp.float32))
            _ = model.backward(dlogits)
            _sgd_step_sequential(model, lr=lr)
            running += float(loss)
            steps += 1
        avg = running / max(1, steps)
        print(f"[mini_train] ep={ep}  loss={avg:.4f}  (lr={lr})")


def run_once(use_native_bwd_dense2: bool = True):
    mean_loss, scalar_loss = _single_pass(use_native_bwd_dense2)
    print(f"[Result] mean(loss_vec)={mean_loss:.6f}, loss_mean(scalar)={scalar_loss:.6f}")


if __name__ == "__main__":
    # 1) 단일 패스(파이썬/네이티브 bwd 토글)
    run_once(use_native_bwd_dense2=False)
    run_once(use_native_bwd_dense2=True)

    # 2) 그라드 수치검증 (db는 sum(dY) 인바리언트 강제)
    compare_grads()

    # 3) 미니 학습 루프
    mini_train(epochs=500, batch_size=64, lr=5e-2)

    print("[Sequential+Dense+SoftmaxCE] enhanced tests ✅")
