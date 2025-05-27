from dev.tests.test_setup import import_cuda_module
import numpy as np
import cupy as cp

# ✅ CUDA 모듈 로드
optimizers_cuda = import_cuda_module(
    module_name="optimizers_cuda",
    build_dir="C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/optimizers/build/lib.win-amd64-cpython-312"
)

# ✅ 옵티마이저 선택 함수
def get(name, learning_rate=0.01):
    if name == "sgd":
        return SGD(learning_rate)
    elif name == "momentum":
        return MomentumSGD(learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

# ✅ SGD Optimizer (in-place CUDA 연산)
class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, weights, dW, bias, db):
        optimizers_cuda.sgd_update(weights, dW, bias, db, self.lr)

# ✅ Momentum SGD Optimizer (in-place CUDA 연산)

class MomentumSGD:
    def __init__(self, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

    def update(self, weights, dW, bias, db):
        if self.v_w is None:
            self.v_w = cp.zeros_like(weights)
            self.v_b = cp.zeros_like(bias)

        optimizers_cuda.momentum_update(
            weights, dW, bias, db,
            self.v_w, self.v_b,
            self.lr, self.momentum
        )
