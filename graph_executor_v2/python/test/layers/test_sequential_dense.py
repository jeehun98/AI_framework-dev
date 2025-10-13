# python/test/layers/test_sequential_dense.py
# --- add project root to sys.path (Windows/any) ---
import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__))                   # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))              # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy


# -------------------- utils --------------------
def max_err(a: cp.ndarray, b: cp.ndarray) -> float:
    return float(cp.max(cp.abs(a - b)))


def rel_err(a: cp.ndarray, b: cp.ndarray, eps: float = 1e-6) -> float:
    num = cp.max(cp.abs(a - b))
    den = cp.maximum(cp.max(cp.abs(a)), cp.max(cp.abs(b)))
    return float(num / (den + eps))


def clone_sequential(src: Sequential, use_native_bwd: bool) -> Sequential:
    """
    src와 동일한 구조/파라미터를 갖는 Sequential을 만든다.
    (현재 Dense만 고려 — 필요시 Conv2D 등 추가)
    """
    dst_layers = []
    for lyr in src.layers:
        if isinstance(lyr, Dense):
            d = Dense(
                units=lyr.units,
                activation=lyr.activation,
                initializer="zeros",  # 곧 덮어씀
                use_native_bwd=use_native_bwd,
                name=(lyr.name + "_clone" if lyr.name else None),
                leaky_slope=lyr.leaky_slope,
            )
            dst_layers.append(d)
        else:
            raise NotImplementedError(f"clone_sequential: unsupported layer type {type(lyr)}")
    dst = Sequential(*dst_layers, name=(src.name + "_clone" if src.name else None))
    dst.build(src.input_shape)

    # 파라미터 복사
    for l_src, l_dst in zip(src.layers, dst.layers):
        assert isinstance(l_src, Dense) and isinstance(l_dst, Dense)
        l_dst.W[...] = l_src.W
        l_dst.b[...] = l_src.b
    return dst


# -------------------- tests --------------------
def run_pair_test_sequential(
    M=64, in_dim=128, hid=256, out_dim=11,
    act_hidden="relu",
    seed=0,
    atol=2e-5, rtol=2e-4, verbose=True,
):
    cp.random.seed(seed)

    # 가짜 분류 데이터
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)

    # 손실
    criterion = SoftmaxCrossEntropy()

    # 모델(수동 backward 버전)
    model_manual = Sequential(
        Dense(hid, activation=act_hidden, initializer="he",     use_native_bwd=False, name="dense1"),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=False, name="dense2"),
        name="MLP_manual",
    )
    model_manual.build((M, in_dim))

    # 네이티브 backward 버전 (동일 파라미터로 복제)
    model_native = clone_sequential(model_manual, use_native_bwd=True)

    # ---------- forward ----------
    model_manual.train(True); model_native.train(True)
    logits0 = model_manual(X)
    logits1 = model_native(X)

    e_y_abs = max_err(logits0, logits1)
    e_y_rel = rel_err(logits0, logits1)
    if verbose:
        print(f"[Seq] forward OK: shape={logits0.shape}, abs={e_y_abs:.3e}, rel={e_y_rel:.3e}")
    assert logits0.shape == (M, out_dim) and logits1.shape == (M, out_dim)
    assert (e_y_abs < atol) or (e_y_rel < rtol), f"forward mismatch: abs={e_y_abs}, rel={e_y_rel}"

    # ---------- loss + grad wrt logits ----------
    loss0, dY0 = criterion.forward(logits0, y)
    loss1, dY1 = criterion.forward(logits1, y)
    if verbose:
        print(f"[Seq] loss(manual={float(loss0):.6f}, native={float(loss1):.6f})")

    # ---------- backward ----------
    model_manual.zero_grad()
    model_native.zero_grad()
    model_manual.backward(dY0)
    model_native.backward(dY1)

    # 파라미터/그라드 수집 (Dense만 가정)
    W0_1, b0_1 = model_manual.layers[0].W, model_manual.layers[0].b
    W0_2, b0_2 = model_manual.layers[1].W, model_manual.layers[1].b
    dW0_1, db0_1 = model_manual.layers[0].dW, model_manual.layers[0].db
    dW0_2, db0_2 = model_manual.layers[1].dW, model_manual.layers[1].db

    W1_1, b1_1 = model_native.layers[0].W, model_native.layers[0].b
    W1_2, b1_2 = model_native.layers[1].W, model_native.layers[1].b
    dW1_1, db1_1 = model_native.layers[0].dW, model_native.layers[0].db
    dW1_2, db1_2 = model_native.layers[1].dW, model_native.layers[1].db

    # 형상/None 체크
    for tag, t in {
        "dW0_1": dW0_1, "db0_1": db0_1, "dW0_2": dW0_2, "db0_2": db0_2,
        "dW1_1": dW1_1, "db1_1": db1_1, "dW1_2": dW1_2, "db1_2": db1_2,
    }.items():
        assert t is not None, f"{tag} is None"

    # ---------- compare grads ----------
    def pair(tag, a, b):
        ea = max_err(a, b); er = rel_err(a, b)
        if verbose:
            print(f"  {tag:<6} abs={ea:.3e}, rel={er:.3e}, shape={a.shape}")
        assert (ea < atol) or (er < rtol), f"{tag} mismatch: abs={ea}, rel={er}"
        return ea, er

    if verbose:
        print("[Seq] backward grads compare (manual vs native)")

    e_W1 = pair("W1", dW0_1, dW1_1)
    e_b1 = pair("b1", db0_1, db1_1)
    e_W2 = pair("W2", dW0_2, dW1_2)
    e_b2 = pair("b2", db0_2, db1_2)

    return {
        "forward_abs": e_y_abs, "forward_rel": e_y_rel,
        "dW1_abs": e_W1[0], "dW1_rel": e_W1[1],
        "db1_abs": e_b1[0], "db1_rel": e_b1[1],
        "dW2_abs": e_W2[0], "dW2_rel": e_W2[1],
        "db2_abs": e_b2[0], "db2_rel": e_b2[1],
        "loss_manual": float(loss0), "loss_native": float(loss1),
    }


def run_capture_step_test(
    M=64, in_dim=128, hid=256, out_dim=11,
    seed=123, steps=30, lr=7.5e-4, wd=5e-5, verbose=True,
):
    """
    Sequential의 plan_capture() / record_graph_step() 기반으로
    CUDA Graph 학습 1스텝을 캡처하고 몇 번 실행해 수렴/정상동작을 점검.
    (Dense.forward_into/backward_into 구현이 필요)
    """
    cp.random.seed(seed)

    # 데이터
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)

    # 모델 (네이티브 BWD로)
    model = Sequential(
        Dense(hid, activation="relu", initializer="he",     use_native_bwd=True, name="dense1"),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True, name="dense2"),
        name="MLP_capture",
    )
    model.build((M, in_dim))
    model.train(True)

    # 손실
    criterion = SoftmaxCrossEntropy()

    # 옵티마이저 스텝 함수(AdamW)
    d1, d2 = model.layers  # type: ignore
    mW1 = cp.zeros_like(d1.W); vW1 = cp.zeros_like(d1.W)
    mB1 = cp.zeros_like(d1.b); vB1 = cp.zeros_like(d1.b)
    mW2 = cp.zeros_like(d2.W); vW2 = cp.zeros_like(d2.W)
    mB2 = cp.zeros_like(d2.b); vB2 = cp.zeros_like(d2.b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = cp.array(0, dtype=cp.int32)  # 타임스텝 (0-D device scalar)

    def adamw_step(W, dW, m, v):
        m[:] = beta1 * m + (1 - beta1) * dW
        v[:] = beta2 * v + (1 - beta2) * (dW * dW)

        tt = t + 1  # 1부터 시작
        # cp.power 사용이 보다 안전 (장치 스칼라 호환)
        bc1 = 1.0 - cp.power(cp.float32(beta1), tt)
        bc2 = 1.0 - cp.power(cp.float32(beta2), tt)

        m_hat = m / bc1
        v_hat = v / bc2

        W[:] = W - lr * (m_hat / (cp.sqrt(v_hat) + eps) + wd * W)

    def optimizer_step_fn():
        # (필요 시) backward_into가 add 누적이라면 아래 주석 해제
        # d1.dW.fill(0); d1.db.fill(0)
        # d2.dW.fill(0); d2.db.fill(0)

        # Dense2
        adamw_step(d2.W, d2.dW, mW2, vW2)
        d2.b[:] = d2.b - lr * d2.db  # bias에는 wd 제거
        # Dense1
        adamw_step(d1.W, d1.dW, mW1, vW1)
        d1.b[:] = d1.b - lr * d1.db

        # !!! 핵심 수정: 0-D 배열 증분은 t += 1 또는 t[...] = t + 1 사용
        t[...] = t + 1


    # 캡처 플랜 & 그래프 레코드
    assert model.supports_capture(), "All layers must implement forward_into/backward_into for capture"
    plan = model.plan_capture((M, in_dim), loss_kind="softmax_ce", lt_bytes=(8 << 20))
    gexec = model.record_graph_step(X, y, loss_fn=criterion, optimizer_step_fn=optimizer_step_fn, capture_plan=plan)

    # 실행
    stream = cp.cuda.Stream(non_blocking=True)
    last_loss = None
    for step in range(1, steps + 1):
        gexec.launch(stream.ptr)
        stream.synchronize()

        logits = plan["buffers"]["fwd"][-1]["y"]
        loss, _ = criterion.forward(logits, y)
        loss_v = float(loss)
        if verbose and step % 10 == 0:
            print(f"[CAP] step={step:03d} loss={loss_v:.6f}")
        last_loss = loss_v

    assert last_loss is not None
    return {"loss_last": last_loss}


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=2e-5)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--no-capture", action="store_true", help="그래프 캡처 테스트 생략")
    args = ap.parse_args()

    stats = run_pair_test_sequential(
        M=64, in_dim=128, hid=256, out_dim=11,
        seed=args.seed, atol=args.atol, rtol=args.rtol, verbose=True,
    )
    print("[SeqDense] manual vs native:", stats)

    if not args.no_capture:
        cap = run_capture_step_test(
            M=64, in_dim=128, hid=256, out_dim=11,
            seed=args.seed + 123, steps=40, verbose=True,
        )
        print("[SeqDense][CAP] ", cap)

    print("[SeqDense] all good ✅")


if __name__ == "__main__":
    main()
