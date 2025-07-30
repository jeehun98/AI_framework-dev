# ✅ CuPy 기반 metrics 정의
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

# ✅ CuPy 기반 wrapper 함수 정의
def mse(y_true, y_pred):
    y_true = cp.asarray(y_true, dtype=cp.float32)
    y_pred = cp.asarray(y_pred, dtype=cp.float32)
    return float(metrics_cuda.mse(y_true, y_pred))

def accuracy(y_true, y_pred):
    y_true = cp.asarray(y_true, dtype=cp.float32)
    y_pred = cp.asarray(y_pred, dtype=cp.float32)
    return float(metrics_cuda.accuracy(y_true, y_pred))

# ✅ Dictionary 및 get 함수 정의
ALL_METRICS_DICT = {
    "mse": mse,
    "accuracy": accuracy,
}

def get(identifier):
    if isinstance(identifier, str):
        identifier = identifier.lower()
        metric_fn = ALL_METRICS_DICT.get(identifier)
        if callable(metric_fn):
            return metric_fn
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Invalid metric identifier: '{identifier}'. "
        f"Available: {', '.join(ALL_METRICS_DICT.keys())}."
    )
