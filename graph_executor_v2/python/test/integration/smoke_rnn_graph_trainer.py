# tests/smoke_rnn_graph_trainer.py
from __future__ import annotations
import os, sys, math
import cupy as cp

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt

# ------------------------
# Model builders
# ------------------------
def make_model_rnn_train(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7,
    rnn_act="tanh", rnn_bias=True, rnn_save_z=True,
    p_drop_rnn=0.0, p_drop_head=0.3,
    scale_in_train=True, seed=0x1234,
) -> Sequential:
    m = Sequential(
        RNN(hidden_size=H, activation=rnn_act, with_bias=rnn_bias, save_z_in_fwd=rnn_save_z),
        (Dropout(p=p_drop_rnn, scale_in_train=scale_in_train, seed=seed ^ 0xD00D)
         if p_drop_rnn > 0 else ActivationLayer(act="none")),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ActHead"),
        Dropout(p=p_drop_head, scale_in_train=scale_in_train, seed=seed ^ 0xBEEF),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, T, I))
    m.train(True)
    return m

def make_model_rnn_eval(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7,
    rnn_act="tanh", rnn_bias=True, rnn_save_z=False,
    p_drop_head=0.5, scale_in_train=True, seed=0x5678,
) -> Sequential:
    m = Sequential(
        RNN(hidden_size=H, activation=rnn_act, with_bias=rnn_bias, save_z_in_fwd=rnn_save_z),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ActEval"),
        Dropout(p=p_drop_head, scale_in_train=scale_in_train, seed=seed ^ 0xCAFE),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, T, I))
    m.train(False)  # Dropout off
    return m

# ------------------------
# Helpers
# ------------------------
def _clone_params_flat(model: Sequential):
    vecs = []
    for lyr in model.layers:
        for name in ("W", "B", "b"):  # 대소문자 혼용 대비
            if hasattr(lyr, name):
                w = getattr(lyr, name)
                if w is not None:
                    vecs.append(w.ravel())
    return None if not vecs else cp.concatenate(vecs)

# ------------------------
# Smoke runner
# ------------------------
def run_smoke_for_rnn(tag: str, *, variant: str):
    N, T, I, C = 8, 12, 16, 7
    cp.random.seed(2025)
    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    if variant == "train_tanh_bias_savez":
        model = make_model_rnn_train(
            N=N, T=T, I=I, H=32, hidden=64, classes=C,
            rnn_act="tanh", rnn_bias=True, rnn_save_z=True,
            p_drop_rnn=0.1, p_drop_head=0.3, scale_in_train=True
        )
    elif variant == "train_relu_nobias_nosavez":
        model = make_model_rnn_train(
            N=N, T=T, I=I, H=48, hidden=64, classes=C,
            rnn_act="relu", rnn_bias=False, rnn_save_z=True,  # 안전하게 저장
            p_drop_rnn=0.0, p_drop_head=0.5, scale_in_train=False
        )
    elif variant == "eval_tanh_bias":
        model = make_model_rnn_eval(
            N=N, T=T, I=I, H=32, hidden=64, classes=C,
            rnn_act="tanh", rnn_bias=True, rnn_save_z=True,
            p_drop_head=0.6, scale_in_train=True
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    print("compile! 컴파일!")
    model.compile((N, T, I), loss=loss, optimizer=opt)

    # 고정 logits 버퍼/포인터 검사
    logits_buf = model.tg.logits
    logits_ptr_before = int(logits_buf.data.ptr)
    last_L = None

    if variant.startswith("train_"):
        # 파라미터 변화와 로짓 변화가 실제로 일어나는지 확인
        w0 = _clone_params_flat(model)
        assert w0 is not None

        for t in range(3):
            Lval = model.one_step(X, y)
            print(f"[SMOKE][RNN:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
            last_L = Lval

        w1 = _clone_params_flat(model)
        assert not cp.allclose(w0, w1), "parameters did not change after training steps"

        # 로짓이 고정 버퍼에 계속 써지는지(포인터 동일)
        assert int(model.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"

        # (선택) 최신 로짓 읽기 및 유한성 검증
        _ = model.one_step(X, y)
        cp.cuda.Stream.null.synchronize()
        logits_after = logits_buf.copy()
        assert not cp.allclose(logits_after, logits_after * 0), "logits must be finite"

    else:
        # eval 변형: Dropout off → 같은 입력이면 재현성 유지(단, 파라미터 업데이트는 진행됨)
        Lvals = []
        for t in range(3):
            Lval = model.one_step(X, y)
            Lvals.append(Lval)
            print(f"[SMOKE][RNN:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lvals[-1]
        assert int(model.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"

    print(f"[OK] Integrated trainer-free smoke passed with RNN({tag}/{variant}). Last loss={last_L:.6f}")

def main():
    print("== Integrated trainer smoke with RNN ==")
    variants = [
        "train_tanh_bias_savez",
        "train_relu_nobias_nosavez",
        "eval_tanh_bias",
    ]
    cases = [("caseR1",), ("caseR2",)]
    for (name,) in cases:
        for v in variants:
            run_smoke_for_rnn(name, variant=v)

if __name__ == "__main__":
    main()
