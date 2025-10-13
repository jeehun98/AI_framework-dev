import os, sys, math
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.pad import Pad
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer


def make_model_dropout_base(
    *,
    N=8, Cin=3, H=16, W=16, hidden=64, classes=7,
    p1=0.1, p2=0.3, scale_in_train=True, seed=0x1234
) -> Sequential:
    """
    Pad → Conv → ReLU → Dropout(p1) → Flatten → Dense → ReLU → Dropout(p2) → Dense
    - Dropout은 파라미터 없음. 캡처 시 플래너가 mask를 준비해줌.
    """
    m = Sequential(
        Pad(before=(1, 1), after=(1, 1), value=0.0),
        Conv2D(out_channels=8, kernel_size=3, padding=(0, 0), activation="none"),
        ActivationLayer(act="relu", save_y=True, name="Act1"),
        Dropout(p=p1, scale_in_train=scale_in_train, seed=seed),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="Act2"),
        Dropout(p=p2, scale_in_train=scale_in_train, seed=seed ^ 0xBEEF),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, Cin, H, W))
    m.train(True)  # 학습 모드 (드롭아웃 on)
    return m


def make_model_dropout_eval(
    *,
    N=8, Cin=3, H=16, W=16, hidden=64, classes=7,
    p=0.5, scale_in_train=True, seed=0x5678
) -> Sequential:
    """
    동일 구조지만 eval 모드에서 캡처/실행(드롭아웃 off, 항등 경로).
    """
    m = Sequential(
        Conv2D(out_channels=8, kernel_size=3, padding=(1, 1), activation="none"),
        ActivationLayer(act="relu", save_y=True, name="ActE1"),
        Dropout(p=p, scale_in_train=scale_in_train, seed=seed),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ActE2"),
        Dropout(p=p, scale_in_train=scale_in_train, seed=seed ^ 0xCAFE),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, Cin, H, W))
    m.train(False)  # 평가 모드 (p는 유지되지만 내부에서 p=0 처리)
    return m


def run_smoke_for_dropout(tag: str, *, variant: str):
    N, Cin, H, W, C = 8, 3, 16, 16, 7
    cp.random.seed(2042)
    X = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    if variant == "train_p01_p03_scale":
        model = make_model_dropout_base(
            N=N, Cin=Cin, H=H, W=W, hidden=64, classes=C,
            p1=0.1, p2=0.3, scale_in_train=True
        )
    elif variant == "train_p05_noscale":
        model = make_model_dropout_base(
            N=N, Cin=Cin, H=H, W=W, hidden=64, classes=C,
            p1=0.5, p2=0.5, scale_in_train=False
        )
    elif variant == "eval_off":
        model = make_model_dropout_eval(
            N=N, Cin=Cin, H=H, W=W, hidden=64, classes=C,
            p=0.6, scale_in_train=True
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, Cin, H, W))  # CUDA Graph capture

    last_L = None
    for t in range(3):
        Lval = trainer.one_step(X, y)
        print(f"[SMOKE][Dropout:{tag}/{variant}] step={t:02d} loss={Lval:.6f}")
        assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lval

    print(f"[OK] Integrated trainer smoke passed with Dropout({tag}/{variant}). Last loss={last_L:.6f}")


def main():
    print("== Integrated trainer smoke with Dropout inserted ==")
    variants = ["train_p01_p03_scale", "train_p05_noscale", "eval_off"]
    cases = [
        ("caseA",),
        ("caseB",),
    ]
    for (name,) in cases:
        for v in variants:
            run_smoke_for_dropout(name, variant=v)


if __name__ == "__main__":
    main()
