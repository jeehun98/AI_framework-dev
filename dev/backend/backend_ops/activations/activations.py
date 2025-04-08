# dev/backend/backend_ops/activations/activations.py

import os
import sys
import numpy as np

# ✅ .pyd 파일이 있는 경로 등록
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

# ✅ CUDA 기반 .pyd 모듈 로드
try:
    import activations_cuda
except ImportError as e:
    raise ImportError(f"❌ activations_cuda import 실패: {e}")

# ✅ 래퍼 함수 정의
def relu(x): return activations_cuda.apply_activation(x, "relu")
def sigmoid(x): return activations_cuda.apply_activation(x, "sigmoid")
def tanh(x): return activations_cuda.apply_activation(x, "tanh")
def leaky_relu(x, alpha=0.01): return activations_cuda.apply_activation(x, "leaky_relu")
def softmax(x): return activations_cuda.apply_activation(x, "softmax")
