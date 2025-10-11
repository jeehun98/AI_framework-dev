import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math
import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.pad import Pad
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer


def make_model(N=8, Cin=3, H=16, W=16, hidden=32, classes=5):
    # 모든 레이어 activation을 'none'으로 설정 (옵션 A: Z==Y alias 경로 검증)
    m = Sequential(
        Pad(before=(1, 1), after=(1, 1), value=0.0),
        Conv2D(out_channels=8, kernel_size=3, padding=(0, 0), activation="none"),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, Cin, H, W))
    m.train(True)
    return m


def main():
    N, Cin, H, W, C = 8, 3, 16, 16, 5
    cp.random.seed(1234)
    X = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    model = make_model(N, Cin, H, W, hidden=32, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, Cin, H, W))

    for t in range(3):
        L = trainer.one_step(X, y)
        print(f"[SMOKE] step={t:02d} loss={L:.6f}")
        assert isinstance(L, float) and math.isfinite(L)

    print("[OK] CUDA Graph capture & run smoke test passed.")


if __name__ == "__main__":
    main()
