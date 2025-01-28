import sys
import os

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
    import matrix_ops
except ImportError as e:
    raise ImportError("Failed to import `matrix_ops` module. Ensure it is built and the path is correctly set.") from e


import numpy as np

# 테스트 데이터 생성
input_data = np.random.rand(4, 4).astype(np.float32)
weights = np.random.rand(4, 4).astype(np.float32)
result = np.zeros((4, 4), dtype=np.float32)

matrix_ops.matrix_mul(input_data, weights, result)

print(result)

bias = 1

bias_reshaped = np.tile(bias, (input_data.shape[0], 1))

# 제대로 동작하는 것을 확인했음
matrix_ops.matrix_add(result, bias_reshaped, result)

print(result)