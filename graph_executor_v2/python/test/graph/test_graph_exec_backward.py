import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.layers.base import Layer
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.linear import Linear
from graph_executor_v2.graph.graph_executor import GraphCompiler

class ReLU(Layer):
    def call(self, x):
        return cp.maximum(x, 0)
    def build(self, input_shape):
        super().build(input_shape); self.output_shape = tuple(int(v) for v in input_shape)
    def compute_output_shape(self, input_shape):
        return tuple(int(v) for v in input_shape)

class Flatten(Layer):
    def call(self, x):
        n = int(x.shape[0]); return x.reshape(n, -1)
    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 4: raise ValueError(f"Flatten expects 4D, got {input_shape}")
        n,c,h,w = map(int, input_shape); self.output_shape = (n, c*h*w)
    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4: raise ValueError(f"Flatten expects 4D, got {input_shape}")
        n,c,h,w = map(int, input_shape); return (n, c*h*w)

class MiniModel:
    def __init__(self):
        self.layers = [
            Conv2D(filters=8, kernel_size=(3,3), stride=(1,1), padding=(1,1), use_bias=True),
            ReLU(),
            Flatten(),
            Linear(out_features=10, use_bias=True),
        ]
    def build(self, input_shape):
        cur = tuple(int(v) for v in input_shape)
        for l in self.layers:
            l.build(cur); cur = l.compute_output_shape(cur)
        self.output_shape = cur

def finite_check(x): return bool(cp.isfinite(x).all())

if __name__ == "__main__":
    cp.random.seed(0)

    x = cp.random.randn(4, 3, 32, 32).astype(cp.float32)
    model = MiniModel(); model.build(x.shape)
    ge = GraphCompiler(model).compile(input_shape=x.shape, use_cuda_graph=False)

    # forward 1
    y = ge.run(x).copy()
    print("[forward] y.shape:", y.shape, "max:", float(y.max()), "norm:", float(cp.linalg.norm(y)))
    t = cp.zeros_like(y)
    loss0 = float(cp.mean((y - t) ** 2))
    print(f"[loss] initial={loss0:.9f}")

    # dL/dY
    gy = (2.0 / y.size) * (y - t)

    # backward (이젠 conv2d까지 연결됨)
    gx = ge.backward(gy)
    assert gx.shape == x.shape
    print("[backward] gx norm:", float(cp.linalg.norm(gx)))

    # grad 유효성
    conv = next(l for l in model.layers if isinstance(l, Conv2D))
    lin  = next(l for l in model.layers if isinstance(l, Linear))

    for name, layer in [("Conv2D", conv), ("Linear", lin)]:
        assert hasattr(layer, "dW") and layer.dW is not None, f"{name}.dW missing"
        if hasattr(layer, "db"):
            assert layer.db is not None, f"{name}.db missing"
        assert finite_check(layer.dW), f"{name}.dW non-finite"
        if hasattr(layer, "db"):
            assert finite_check(layer.db), f"{name}.db non-finite"

    # 러닝레이트 스윕으로 손실 감소하는 lr 선택
    candidate_lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    best = (None, loss0)

    # 현재 파라미터 스냅샷
    convW0, convb0 = conv.W.copy(), conv.b.copy()
    linW0,  linb0  = lin.W.copy(),  lin.b.copy()

    for lr in candidate_lrs:
        # 임시 적용
        conv.W[...] = convW0 - lr * conv.dW
        conv.b[...] = convb0 - lr * conv.db
        lin .W[...] = linW0  - lr * lin .dW
        lin .b[...] = linb0  - lr * lin .db

        y_try = ge.run(x).copy()
        loss_try = float(cp.mean((y_try - t) ** 2))
        print(f"[try] lr={lr:.1e} loss={loss_try:.9f}")

        if loss_try < best[1]:
            best = (lr, loss_try)

    # 최적 lr 적용 (없으면 아주 작은 lr 사용)
    for p, q in [(conv.W, convW0), (conv.b, convb0), (lin.W, linW0), (lin.b, linb0)]:
        p[...] = q  # 리셋
    lr = best[0] if best[0] is not None else 1e-3

    conv.W[...] = convW0 - lr * conv.dW
    conv.b[...] = convb0 - lr * conv.db
    lin .W[...] = linW0  - lr * lin .dW
    lin .b[...] = linb0  - lr * lin .db

    y2 = ge.run(x).copy()
    loss1 = float(cp.mean((y2 - t) ** 2))
    print(f"[choose] lr={lr:.1e}")
    print(f"[loss] before={loss0:.9f} after_update={loss1:.9f}")
    print("[delta] ||y2-y||:", float(cp.linalg.norm(y2 - y)))
