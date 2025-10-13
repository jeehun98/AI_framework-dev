# python/examples/train_mlp_sequential.py
from __future__ import annotations
import os, sys
import math

# --- Import path (repo/python) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)


import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamW  # 또는 SGD
# from graph_executor_v2.optim.sgd import SGD

def main():
    cp.random.seed(0)

    # 하이퍼파라미터
    batch = 128
    in_dim = 256
    hid    = 512
    out_dim = 10
    lr = 1e-3
    steps = 200

    # 가짜 데이터(분류)
    X = cp.random.randn(batch, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(batch,), dtype=cp.int32)

    # 모델
    model = Sequential(
        Dense(hid, activation="relu", initializer="he", use_native_bwd=True),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True),
        name="MLP_2xDense"
    )
    model.build((batch, in_dim))
    print(model.summary())

    # 손실/옵티마이저
    criterion = SoftmaxCrossEntropy()  # reduction='mean' or 'sum'는 내부 디폴트에 따름
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # opt = SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train(True)

    for step in range(1, steps+1):
        # ---- forward
        logits = model(X)                         # (batch, out_dim)

        # ---- loss
        loss, grad_logits = criterion.forward_with_grad(logits, y)  # (scalar), (batch, out_dim)

        # ---- backward
        model.zero_grad()
        model.backward(grad_logits)
        model.attach_grads()                      # 파라미터 객체의 .grad에 연결

        # ---- optimize
        opt.step()
        opt.zero_grad()                           # 옵티마이저 grad 뒷정리

        if step % 20 == 0:
            # 간단 정확도
            pred = logits.argmax(axis=1)
            acc = float((pred == y).mean())
            print(f"[{step:04d}] loss={float(loss):.4f}  acc={acc:.3f}")

    print("done.")

if __name__ == "__main__":
    main()
