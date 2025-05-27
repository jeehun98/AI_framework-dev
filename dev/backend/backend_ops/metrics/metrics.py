import os
import sys
import cupy as cp

# ✅ .pyd 경로 등록
pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))
if pyd_path not in sys.path:
    sys.path.insert(0, pyd_path)

# ✅ CUDA DLL 경로 등록
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_path)

# ✅ Pybind 모듈 import
import metrics_cuda

# CuPy 기반 wrapper 함수
def mse(y_true, y_pred):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    return float(metrics_cuda.mse(y_true, y_pred))

def accuracy(y_true, y_pred):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    return float(metrics_cuda.accuracy(y_true, y_pred))
