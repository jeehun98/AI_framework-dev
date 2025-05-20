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
    import losses_cuda
except ImportError as e:
    raise ImportError(f"❌ losses_cuda import 실패: {e}")

# ============================================
# 🚨 CUDA 연산 래퍼 함수
# ============================================

def mse(y_true, y_pred):
    return losses_cuda.mse_loss(y_true.astype(np.float32), y_pred.astype(np.float32))

def binary_crossentropy(y_true, y_pred):
    return losses_cuda.binary_crossentropy(y_true.astype(np.float32), y_pred.astype(np.float32))

def categorical_crossentropy(y_true, y_pred):
    raise NotImplementedError("Categorical Crossentropy는 아직 CUDA에 구현되지 않았습니다.")
ALL_LOSSES_DICT = {
    "mse": mse,
    "binary_crossentropy": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy,
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
        f"Available options: {', '.join(ALL_LOSSES_DICT.keys())}."
    )
