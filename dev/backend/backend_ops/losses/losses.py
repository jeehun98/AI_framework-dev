import os
import sys
import cupy as cp  # ✅ CuPy 기반
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
    import losses_cuda
except ImportError as e:
    raise ImportError(f"❌ losses_cuda import 실패: {e}")

# ============================================
# 🚀 Forward 연산 (loss) 래퍼
# ============================================

import cupy as cp

def mse(y_true, y_pred):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    return float(losses_cuda.mse_loss(y_true, y_pred))


def binary_crossentropy(y_true, y_pred):
    return float(losses_cuda.binary_crossentropy(y_true, y_pred))

def categorical_crossentropy(y_true, y_pred):
    return float(losses_cuda.categorical_crossentropy(y_true, y_pred))

# ============================================
# 🔁 Backward (gradient) 연산 래퍼
# ============================================
def mse_grad(y_true, y_pred):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    grad = cp.zeros_like(y_true)
    losses_cuda.mse_grad(y_true, y_pred, grad)
    return grad

def bce_grad(y_true, y_pred):
    grad = cp.zeros_like(y_true)
    losses_cuda.bce_grad(y_true, y_pred, grad)
    return grad

def cce_grad(y_true, y_pred):
    grad = cp.zeros_like(y_true)
    losses_cuda.cce_grad(y_true, y_pred, grad)
    return grad

# ============================================
# 📦 Dict 등록 및 Getter
# ============================================

ALL_LOSSES_DICT = {
    "mse": mse,
    "binary_crossentropy": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy,
}

ALL_LOSSES_GRAD_DICT = {
    "mse": mse_grad,
    "binary_crossentropy": bce_grad,
    "categorical_crossentropy": cce_grad,
}

def get(identifier):
    if isinstance(identifier, str):
        identifier = identifier.lower()
        loss_fn = ALL_LOSSES_DICT.get(identifier)
        if callable(loss_fn):
            return loss_fn
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Invalid loss function identifier: '{identifier}'. "
        f"Available: {', '.join(ALL_LOSSES_DICT.keys())}."
    )

def get_grad(identifier):
    if isinstance(identifier, str):
        identifier = identifier.lower()
        grad_fn = ALL_LOSSES_GRAD_DICT.get(identifier)
        if callable(grad_fn):
            return grad_fn
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Invalid loss gradient identifier: '{identifier}'. "
        f"Available: {', '.join(ALL_LOSSES_GRAD_DICT.keys())}."
    )
