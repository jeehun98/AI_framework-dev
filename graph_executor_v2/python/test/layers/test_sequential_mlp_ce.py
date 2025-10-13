# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))  # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))  # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp

# 레이어/모델
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.layers.model import Sequential   # 이전 메시지의 Sequential 추가했다고 가정

def _sgd_step_sequential(model: Sequential, lr: float = 1e-2):
    """
    매우 단순한 파라미터 업데이트: p -= lr * g
    (Dense가 W, dW, b, db 속성을 갖는다는 네 테스트 전제를 그대로 사용)
    """
    for lyr in getattr(model, "layers", []):
        for p_name, g_name in [("W", "dW"), ("b", "db")]:
            if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                p, g = getattr(lyr, p_name), getattr(lyr, g_name)
                if g is None:
                    continue
                p[...] = p - lr * g  # CuPy ndarray in-place

def run_once(use_native_bwd_dense2: bool = True):
    cp.random.seed(42)

    # -----------------------------
    # 1) 데이터 생성
    # -----------------------------
    M, Din, H, Cout = 32, 64, 128, 10
    x = cp.random.randn(M, Din).astype(cp.float32)
    y = cp.random.randint(0, Cout, size=(M,), dtype=cp.int32)

    # -----------------------------
    # 2) 모델 구성 (Dense → Dense)
    #    네 Dense 테스트 스타일을 그대로 유지: activation, initializer, use_native_bwd
    # -----------------------------
    model = Sequential(
        Dense(
            units=H,
            activation="relu",
            initializer="he",
            use_native_bwd=False,     # 첫 레이어는 Python bwd 경로
            name="dense1"
        ),
        Dense(
            units=Cout,
            activation=None,          # logits 직접 출력
            initializer="xavier",
            use_native_bwd=use_native_bwd_dense2,
            name="dense2"
        ),
        name="mlp_seq"
    )

    # -----------------------------
    # 3) 순전파 (logits)
    # -----------------------------
    logits = model(x)
    assert logits.shape == (M, Cout), f"logits.shape={logits.shape}"
    print(f"[Sequential] forward OK: logits shape {logits.shape}")

    # -----------------------------
    # 4) 손실 (SoftmaxCrossEntropy)
    #    네 테스트와 동일하게 reduction='none' → per-sample loss,
    #    backward에는 (M,) 스케일 벡터 사용
    # -----------------------------
    ce = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="none", from_logits=True)
    loss_vec = ce((logits, y))
    assert loss_vec.shape == (M,), f"loss_vec.shape={loss_vec.shape}"
    print(f"[SoftmaxCE] forward OK: loss per-sample shape {loss_vec.shape}")

    grad_scale = cp.full((M,), 1.0 / M, dtype=cp.float32)
    dlogits = ce.backward(grad_scale)
    assert dlogits.shape == logits.shape, f"dlogits.shape={dlogits.shape}"
    print("[SoftmaxCE] backward OK (reduction='none')")

    # -----------------------------
    # 5) 역전파 (모델)
    # -----------------------------
    dx = model.backward(dlogits)
    assert dx.shape == x.shape, f"dx.shape={dx.shape}"
    # Dense 레이어들의 그래드 확인 (네 Dense 테스트와 합치)
    for lyr in model.layers:
        if isinstance(lyr, Dense):
            assert lyr.dW is not None and lyr.db is not None
    print(f"[Sequential] backward OK (use_native_bwd_dense2={use_native_bwd_dense2})")

    # -----------------------------
    # 6) (옵션) 간단 SGD 스텝
    # -----------------------------
    _sgd_step_sequential(model, lr=1e-2)
    print("[Optim] SGD step OK")

    # -----------------------------
    # 7) reduction='mean' 경로도 점검 (네 테스트와 동일)
    # -----------------------------
    ce_mean = SoftmaxCrossEntropy(label_smoothing=0.0, reduction="mean", from_logits=True)
    loss_mean = ce_mean((logits, y))
    assert loss_mean.shape == (1,), f"loss_mean.shape={loss_mean.shape}"
    dlogits_mean = ce_mean.backward(cp.array(1.0, dtype=cp.float32))  # scalar scale
    assert dlogits_mean.shape == logits.shape
    print("[SoftmaxCE] reduction='mean' OK")

    # -----------------------------
    # 8) summary 출력 (선택)
    # -----------------------------
    try:
        print(model.summary())
    except Exception:
        # 일부 레이어가 compute_output_shape를 정확히 구현하지 않았으면 ?가 나올 수 있음
        pass

if __name__ == "__main__":
    # 네 Dense 단일 테스트처럼, 다른 backward 경로도 함께 체크
    run_once(use_native_bwd_dense2=False)
    run_once(use_native_bwd_dense2=True)
    print("[Sequential+Dense+SoftmaxCE] all good ✅")
