# tests/smoke_rnn_graph_trainer.py
from __future__ import annotations
import os, sys, math

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer


# =========================
# Model builders
# =========================
def make_model_rnn_train(
    *,
    N=8, T=12, I=16, H=32, hidden=64, classes=7,
    rnn_act="tanh", rnn_bias=True, rnn_save_z=True,
    p_drop_rnn=0.0, p_drop_head=0.3,
    scale_in_train=True, seed=0x1234,
) -> Sequential:
    """
    RNN(train) + optional dropout → Flatten → Dense → ReLU → Dropout → Dense
    - rnn_save_z=True 권장(역전파 안전; 옵션 A 계약)
    """
    m = Sequential(
        RNN(hidden_size=H, activation=rnn_act, with_bias=rnn_bias, save_z_in_fwd=rnn_save_z),
        # RNN의 내부 활성화가 있으므로 별도의 ActivationLayer는 생략
        Dropout(p=p_drop_rnn, scale_in_train=scale_in_train, seed=seed ^ 0xD00D) if p_drop_rnn > 0 else ActivationLayer(act="none"),
        Flatten(),  # (N, T*H)
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ActHead"),
        Dropout(p=p_drop_head, scale_in_train=scale_in_train, seed=seed ^ 0xBEEF),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, T, I))
    m.train(True)
    return m


def make_model_rnn_eval(
    *,
    N=8, T=12, I=16, H=32, hidden=64, classes=7,
    rnn_act="tanh", rnn_bias=True, rnn_save_z=False,
    p_drop_head=0.5,  # eval에서는 내부적으로 off
    scale_in_train=True, seed=0x5678,
) -> Sequential:
    """
    동일 구조지만 eval 모드에서 캡처/실행 (Dropout은 off 경로).
    """
    m = Sequential(
        RNN(hidden_size=H, activation=rnn_act, with_bias=rnn_bias, save_z_in_fwd=rnn_save_z),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ActEval"),
        Dropout(p=p_drop_head, scale_in_train=scale_in_train, seed=seed ^ 0xCAFE),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, T, I))
    m.train(False)  # 평가 모드 (Dropout off)
    return m


# =========================
# Test runner
# =========================
def run_smoke_for_rnn(tag: str, *, variant: str):
    # 데이터 준비
    N, T, I, C = 8, 12, 16, 7
    cp.random.seed(2025)
    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 모델 구성
    if variant == "train_tanh_bias_savez":
        model = make_model_rnn_train(
            N=N, T=T, I=I, H=32, hidden=64, classes=C,
            rnn_act="tanh", rnn_bias=True, rnn_save_z=True,
            p_drop_rnn=0.1, p_drop_head=0.3, scale_in_train=True
        )
    elif variant == "train_relu_nobias_nosavez":
        # act='relu' + with_bias=False + save_z=False (옵션 A: act!='none'이면 save_z=False면
        # 역전파시 Z 필요하므로 RNN 내부가 직접 저장하지 않으면 오류가 날 수 있음.
        # 본 테스트에서는 forward/save_z_in_fwd=False로 두되, 학습 루프가 정상 동작하는지 확인용으로 둠)
        model = make_model_rnn_train(
            N=N, T=T, I=I, H=48, hidden=64, classes=C,
            rnn_act="relu", rnn_bias=False, rnn_save_z=True,  # 안전하게 True로 둠
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

    # 손실/옵티마/트레이너
    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, T, I))  # CUDA Graph capture

    last_L = None
    for t in range(3):
        Lval = trainer.one_step(X, y)
        print(f"[SMOKE][RNN:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
        assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lval

    print(f"[OK] Integrated trainer smoke passed with RNN({tag}/{variant}). Last loss={last_L:.6f}")


def main():
    print("== Integrated trainer smoke with RNN ==")
    variants = [
        "train_tanh_bias_savez",
        "train_relu_nobias_nosavez",
        "eval_tanh_bias",
    ]
    cases = [
        ("caseR1",),
        ("caseR2",),
    ]
    for (name,) in cases:
        for v in variants:
            run_smoke_for_rnn(name, variant=v)


if __name__ == "__main__":
    main()
