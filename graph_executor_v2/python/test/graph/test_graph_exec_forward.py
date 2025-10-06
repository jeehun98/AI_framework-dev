# 예: python/test/graph/test_graph_exec_forward.py
import os, sys

# sys.path
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# python/test/graph/test_graph_exec_forward.py
import cupy as cp
from graph_executor_v2.layers.base import Layer
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.linear import Linear
from graph_executor_v2.graph.graph_executor import GraphCompiler

class ReLU(Layer):
    def call(self, x):
        return cp.maximum(x, 0)

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = tuple(int(v) for v in input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(int(v) for v in input_shape)

class Flatten(Layer):
    def call(self, x):
        n = int(x.shape[0])
        return x.reshape(n, -1)

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 4:
            raise ValueError(f"Flatten expects 4D input (N,C,H,W), got {input_shape}")
        n, c, h, w = map(int, input_shape)
        self.output_shape = (n, c*h*w)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Flatten expects 4D input (N,C,H,W), got {input_shape}")
        n, c, h, w = map(int, input_shape)
        return (n, c*h*w)

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
            l.build(cur)
            cur = l.compute_output_shape(cur)
        self.output_shape = cur

if __name__ == "__main__":
    x = cp.random.randn(4, 3, 32, 32).astype(cp.float32)

    model = MiniModel()
    model.build(x.shape)

    ge = GraphCompiler(model).compile(input_shape=x.shape, use_cuda_graph=True)

    y = ge.run(x)
    print("y.shape:", y.shape, "max:", float(y.max()), "norm:", float(cp.linalg.norm(y)))
