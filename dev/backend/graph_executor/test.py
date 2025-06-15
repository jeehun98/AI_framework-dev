import sys
import os
import ctypes
import numpy as np

# DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))
import graph_executor

# E 행렬 정의
E = np.array([
    [0, 0, -1, 0, -1],
    [1, 0, -2, 1, -1],
    [2, 0, -3, 2, -1],
], dtype=np.int32)

# 입출력 및 파라미터 정의
W = np.random.randn(4, 3).astype(np.float32)
b = np.zeros((1, 3), dtype=np.float32)
x = np.random.randn(1, 4).astype(np.float32)
out = np.zeros((1, 3), dtype=np.float32)

# shape 정보
shapes = np.array([
    E.size,
    x.shape[0],  # batch
    x.shape[1],  # input
    W.shape[0],  # W rows
    W.shape[1],  # W cols
], dtype=np.int32)

# 실행
graph_executor.run_graph_cuda(E, shapes, W, b, x, out)

print("✅ CUDA 그래프 실행 완료!")
print("출력:", out)
