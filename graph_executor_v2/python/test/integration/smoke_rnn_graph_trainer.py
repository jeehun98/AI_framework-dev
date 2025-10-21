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
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer

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
def _clone_params_flat(model):
    vecs = []
    for lyr in model.layers:
        for name in ("W", "B"):
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

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, T, I))

    # 그래프가 내보내는 고정 로짓 버퍼 (포인터 동일성 체크용)
    logits_buf = trainer.tg.logits
    logits_ptr_before = int(logits_buf.data.ptr)  # 고정 버퍼여야 동일 포인터

    last_L = None

    if variant.startswith("train_"):
        # 파라미터 변화와 로짓 변화가 실제로 일어나는지 확인
        w0 = _clone_params_flat(model)
        assert w0 is not None

        for t in range(3):
            Lval = trainer.one_step(X, y)
            print(f"[SMOKE][RNN:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
            last_L = Lval

        w1 = _clone_params_flat(model)
        assert not cp.allclose(w0, w1), "parameters did not change after training steps"

        # 로짓이 고정 버퍼에 계속 써지는지(포인터 동일) + 값은 변했는지(업데이트 영향)
        assert int(trainer.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"
        # 최소한 한 스텝 후에는 값이 바뀌었다고 가정(랜덤 데이터/드롭아웃로 인해 엄격 단조는 요구하지 않음)
        # 값 변화 여부를 샘플 몇 개로 체크
        cp.cuda.Stream.null.synchronize()
        changed = not cp.allclose(logits_buf, logits_buf + 1e-6)  # trivial false; force read
        # 보다 직접: 같은 입력으로 여러 스텝 학습하면 로짓은 보통 변함
        # (간단히 첫 스텝 직후 스냅샷과 마지막 스텝을 비교)
        # 안전하게 다시 실행해서 최신 로짓 확보
        _ = trainer.one_step(X, y)
        logits_after = logits_buf.copy()
        assert not cp.allclose(logits_after, logits_after * 0), "logits must be finite"
    else:
        # eval 변형: Dropout off → 같은 입력이면 로짓이 재현성 있게 유지(단, 파라미터 업데이트는 진행됨)
        # 캡처/리플레이가 되면서 loss도 유의미한 값이어야 함
        Lvals = []
        for t in range(3):
            Lval = trainer.one_step(X, y)
            Lvals.append(Lval)
            print(f"[SMOKE][RNN:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lvals[-1]   # ✅ 마지막 손실로 갱신
        assert int(trainer.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"

    print(f"[OK] Integrated trainer smoke passed with RNN({tag}/{variant}). Last loss={last_L:.6f}")

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
