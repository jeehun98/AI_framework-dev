import numpy as np
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

# 테스트 데이터 생성
A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)
C = np.zeros((4, 4), dtype=np.float32)

print(A)

print(B)

# CUDA 행렬 덧셈
try:
    matrix_ops.matrix_add(C, B, C)
    print("CUDA Addition Result:\n", C)
    # CPU 연산 결과 검증
    assert np.allclose(C, B), "Addition result does not match!"
    print("Addition test passed.")
except Exception as e:
    print("Error in CUDA addition:", e)

# CUDA 행렬 곱셈
try:
    matrix_ops.matrix_mul(C, B, C)
    print("CUDA Multiplication Result:\n", C)
    # CPU 연산 결과 검증
    assert np.allclose(C, np.dot(C, B)), "Multiplication result does not match!"
    print("Multiplication test passed.")
except Exception as e:
    print("Error in CUDA multiplication:", e)
