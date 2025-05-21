import os
import sys
import numpy as np

# ✅ .pyd (또는 .so) 경로 등록
pyd_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))
if pyd_dir not in sys.path:
    sys.path.insert(0, pyd_dir)

# ✅ CUDA DLL 경로 등록 (Windows 환경일 경우)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

# ✅ CUDA 모듈 로드
try:
    import optimizer_cuda
except ImportError as e:
    raise ImportError(f"❌ optimizer_cuda import 실패: {e}")


# ============================
# 🚀 Python Optimizer Wrapper
# ============================

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, dW, bias, db):
        weights = weights.astype(np.float32)
        dW = dW.astype(np.float32)
        bias = bias.astype(np.float32)
        db = db.astype(np.float32)

        optimizer_cuda.sgd_update(weights, dW, self.learning_rate)
        optimizer_cuda.sgd_update(bias, db, self.learning_rate)
        return weights, bias

    def get_config(self):
        return {"optimizer": "SGD", "learning_rate": self.learning_rate}


class MomentumSGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vel_w = None
        self.vel_b = None

    def update(self, weights, dW, bias, db):
        weights = weights.astype(np.float32)
        dW = dW.astype(np.float32)
        bias = bias.astype(np.float32)
        db = db.astype(np.float32)

        if self.vel_w is None:
            self.vel_w = np.zeros_like(weights)
            self.vel_b = np.zeros_like(bias)

        optimizer_cuda.momentum_update(weights, dW, self.vel_w, self.learning_rate, self.momentum)
        optimizer_cuda.momentum_update(bias, db, self.vel_b, self.learning_rate, self.momentum)

        return weights, bias

    def get_config(self):
        return {
            "optimizer": "MomentumSGD",
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        }


# ============================
# 🔍 옵티마이저 생성기
# ============================

def get(identifier, learning_rate=0.01, momentum=0.9):
    if identifier == "sgd":
        return SGD(learning_rate=learning_rate)
    elif identifier == "momentum":
        return MomentumSGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"❌ Unknown optimizer: {identifier}")
