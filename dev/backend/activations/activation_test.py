# backend 를 통한 activation function 연동 테스트트

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# 빌드된 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise FileNotFoundError(f"Build path does not exist: {build_path}")

# CUDA DLL 경로 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    raise FileNotFoundError(f"CUDA path does not exist: {cuda_path}")

try:
    import activation_cuda
except ImportError as e:
    raise ImportError("Failed to import `matrix_ops` module. Ensure it is built and the path is correctly set.") from e

