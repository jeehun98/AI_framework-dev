import os
import sys
import numpy as np

# ✅ .pyd 경로 등록
pyd_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))
if pyd_dir not in sys.path:
    sys.path.insert(0, pyd_dir)

# ✅ CUDA DLL 경로 등록
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

# ✅ CUDA 모듈 로드
try:
    import activations_cuda
except ImportError as e:
    raise ImportError(f"❌ activations_cuda import 실패: {e}")

# ============================================
# 🚨 CUDA 연산 래퍼 함수 (In-place 기반)
# ============================================

def relu(x):
    x = x.astype(np.float32, copy=True)
    activations_cuda.apply_activation(x, "relu")
    return x

def sigmoid(x):
    x = x.astype(np.float32, copy=True)
    activations_cuda.apply_activation(x, "sigmoid")
    return x

def tanh(x):
    x = x.astype(np.float32, copy=True)
    activations_cuda.apply_activation(x, "tanh")
    return x

def leaky_relu(x, alpha=0.01):
    raise NotImplementedError("Leaky ReLU는 아직 CUDA에 구현되지 않았습니다.")

def softmax(x):
    raise NotImplementedError("Softmax는 아직 CUDA에 구현되지 않았습니다.")
