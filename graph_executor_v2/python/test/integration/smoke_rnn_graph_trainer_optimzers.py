# tests/smoke_rnn_graph_trainer_optimizers.py
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
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer

# ⬇️ AdamW 모듈에 있는 수집 유틸을 재사용합니다.
from graph_executor_v2.optim.adamw import AdamWOpt, collect_params_from, collect_params_from_plan

# 선택 가능한 추가 옵티마이저들 (미리 구현되어 있다고 가정)
try:
    from graph_executor_v2.optim.sgd import SGDOpt
except Exception:
    SGDOpt = None
try:
    from graph_executor_v2.optim.rmsprop import RMSpropOpt
except Exception:
    RMSpropOpt = None
try:
    from graph_executor_v2.optim.adagrad import AdagradOpt
except Exception:
    AdagradOpt = None
try:
    from graph_executor_v2.optim.amsgrad import AMSGradOpt
except Exception:
    AMSGradOpt = None
try:
    from graph_executor_v2.optim.lion import LionOpt
except Exception:
    LionOpt = None
try:
    from graph_executor_v2.optim.lamb import LAMBOpt
except Exception:
    LAMBOpt = None

# ------------------------
# Model builders (원본과 동일)
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
        # 주의: Dense/Conv 계열에서 변수명이 'W','B' 또는 'b'일 수 있어 보강
        for name in ("W", "B", "b"):
            if hasattr(lyr, name):
                w = getattr(lyr, name)
                if w is not None:
                    vecs.append(w.ravel())
    return None if not vecs else cp.concatenate(vecs)

def _make_optimizer(opt_name: str, model, trainer_input_shape):
    """
    캡처 전 파라미터를 수집해 옵티마이저 생성 → compile → 실제 bwd 버퍼로 rebind.
    """
    # 1) 캡처 전 수집(placeholder grad 허용)

    # 2) 옵티마이저 생성
    name = opt_name.lower()
    if name == "adamw":
        opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    elif name == "sgd" and SGDOpt is not None:
        opt = SGDOpt([], lr=5e-2, momentum=0.9, nesterov=True, wd=1e-4)
    elif name == "rmsprop" and RMSpropOpt is not None:
        opt = RMSpropOpt([], lr=1e-3, alpha=0.99, eps=1e-8, wd=1e-4)
    elif name == "adagrad" and AdagradOpt is not None:
        opt = AdagradOpt([], lr=5e-2, eps=1e-10, wd=1e-4)
    elif name == "amsgrad" and AMSGradOpt is not None:
        opt = AMSGradOpt([], lr=1e-3, wd=1e-4, beta1=0.9, beta2=0.999, eps=1e-8)
    elif name == "lion" and LionOpt is not None:
        opt = LionOpt([], lr=1e-4, beta1=0.9, wd=1e-2)
    elif name == "lamb" and LAMBOpt is not None:
        opt = LAMBOpt([], lr=1e-3, wd=1e-2, beta1=0.9, beta2=0.999, eps=1e-6)
    else:
        raise ValueError(f"Unknown or unavailable optimizer: {opt_name}")

    # 3) 트레이너 생성 & 컴파일
    trainer = CudaGraphTrainer(model, SoftmaxCrossEntropy(), opt)
    plan = trainer.compile(trainer_input_shape)

    # 4) 캡처 이후 실제 bwd 버퍼로 rebind (트레이너가 내부에서 처리한다면 생략 가능)
    if hasattr(trainer, "capture_plan"):
        opt.rebind_grads(collect_params_from_plan(model, trainer.capture_plan))
    return trainer, opt

# ------------------------
# Smoke runner
# ------------------------
def run_smoke_for_rnn(tag: str, *, variant: str, opt_name: str):
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
            rnn_act="relu", rnn_bias=False, rnn_save_z=True,
            p_drop_rnn=0.0, p_drop_head=0.5, scale_in_train=False
        )
    elif variant == "eval_tanh_bias":
        model = make_model_rnn_eval(
            N=N, T=T, I=I, H=32, hidden=64, classes=C,
            rnn_act="tanh", rnn_bias=True, rnn_save_z=False,
            p_drop_head=0.6, scale_in_train=True
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    trainer, opt = _make_optimizer(opt_name, model, (N, T, I))

    # 고정 로짓 버퍼 포인터(캡처 안정성)
    logits_buf = trainer.tg.logits
    logits_ptr_before = int(logits_buf.data.ptr)

    last_L = None

    if variant.startswith("train_"):
        # 파라미터 변화 검증
        w0 = _clone_params_flat(model)
        assert w0 is not None

        for t in range(4):
            Lval = trainer.one_step(X, y)
            print(f"[SMOKE][RNN:{tag}/{variant}/{opt_name}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
            last_L = Lval

        w1 = _clone_params_flat(model)
        assert not cp.allclose(w0, w1), "parameters did not change after training steps"

        # logits 포인터 고정
        assert int(trainer.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"
        cp.cuda.Stream.null.synchronize()
        # 값 유효성(유한성) 간단 체크
        _ = trainer.one_step(X, y)
        logits_after = logits_buf.copy()
        assert cp.isfinite(logits_after).all(), "logits must be finite"
    else:
        # eval 변형: 드롭아웃 off, 그래프 캡처/리플레이 유효성
        Lvals = []
        for t in range(3):
            Lval = trainer.one_step(X, y)
            Lvals.append(Lval)
            print(f"[SMOKE][RNN:{tag}/{variant}/{opt_name}] step={t:02d} loss={Lval:.6f}")
            assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lvals[-1]
        assert int(trainer.tg.logits.data.ptr) == logits_ptr_before, "logits buffer must be static in capture"

    print(f"[OK] RNN smoke passed: {tag}/{variant} with {opt_name}. Last loss={last_L:.6f}")

def main():
    print("== Integrated trainer smoke with RNN (optimizer mix) ==")
    variants = [
        "train_tanh_bias_savez",
        "train_relu_nobias_nosavez",
        "eval_tanh_bias",
    ]
    # 사용 가능한 옵티마이저만 자동 선택
    OPT_NAMES = ["adamw"]
    if SGDOpt is not None: OPT_NAMES.append("sgd")
    if RMSpropOpt is not None: OPT_NAMES.append("rmsprop")
    if AdagradOpt is not None: OPT_NAMES.append("adagrad")
    if AMSGradOpt is not None: OPT_NAMES.append("amsgrad")
    if LionOpt is not None: OPT_NAMES.append("lion")
    if LAMBOpt is not None: OPT_NAMES.append("lamb")

    cases = [("caseR1",), ("caseR2",)]
    for (name,) in cases:
        for opt_name in OPT_NAMES:
            for v in variants:
                run_smoke_for_rnn(name, variant=v, opt_name=opt_name)

if __name__ == "__main__":
    main()
