import os
import sys
import numpy as np

# CUDA DLL 경로 명시적 추가 (Python 3.8 이상)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_path)
else:
    os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

sys.path.append("build/lib.win-amd64-cpython-312")

# CUDA 확장 모듈 불러오기
try:
    import cuda_add
    print("cuda_add module imported successfully.")
except ImportError as e:
    print("ImportError:", e)

# 테스트 데이터 생성
size = 1024
a = np.ones(size, dtype=np.float32)
b = np.ones(size, dtype=np.float32)
c = np.zeros(size, dtype=np.float32)

# CUDA 함수 호출
cuda_add.vector_add(a, b, c)

# 결과 확인
print("Result:", c[:10])
