from dev.tests.test_setup import import_cuda_module
import numpy as np

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
        self.v_w = None  # weight velocity
        self.v_b = None  # bias velocity

    def update(self, weights, dW, bias, db):
        # 초기화 시점에서 zeros_like 사용
        if self.v_w is None:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(bias)

        # CUDA 커널 실행 (in-place 업데이트)
        optimizers_cuda.momentum_update(
            weights, dW, bias, db,
            self.v_w, self.v_b,
            self.lr, self.momentum
        )
