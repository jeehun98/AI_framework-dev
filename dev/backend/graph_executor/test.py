import sys
import os
import ctypes
import numpy as np

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# 모듈 로드
import graph_executor

# 연산 계획
E = np.array([
    [0, 0, -1, 0, -1],   # matmul
    [1, 0, -2, 1, -1],   # add
    [2, 0, -3, 2, -1],   # relu
], dtype=np.int32)

# 파라미터
W = np.random.randn(4, 3).astype(np.float32)
b = np.zeros((1, 3), dtype=np.float32)

# 입력, 출력
x = np.random.randn(1, 4).astype(np.float32)
out = np.zeros((1, 3), dtype=np.float32)

# shape 정보
shapes = np.array([
    E.size,         # E_len
    x.shape[0],     # batch
    x.shape[1],     # input_dim
    W.shape[0],     # W_rows
    W.shape[1],     # W_cols
], dtype=np.int32)

# 실행
graph_executor.run_graph_cuda(
    E.flatten(), E.size,
    shapes, shapes.size,
    W.flatten(), b.flatten(),
    W.shape[0], W.shape[1]
)
