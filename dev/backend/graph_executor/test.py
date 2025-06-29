import sys
import os
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import numpy as np
import graph_executor

# === 설정 ===
batch = 2         # 입력 샘플 수 (행 개수)
input_dim = 3     # 입력 특징 차원
output_dim = 4    # 출력 차원 (가중치 열 수)

# === E와 shapes는 아직 실제로 사용되지 않지만 구조 맞춰서 전달 ===
# E: dummy 그래프 정의 배열
E = np.array([0], dtype=np.int32)

# shapes: [num_dims, batch, input_dim] 구조라고 가정 (run_graph_cuda 내부에서 사용)
shapes = np.array([3, batch, input_dim], dtype=np.int32)

# === 가중치 및 편향 ===
W = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5, 0.5]
], dtype=np.float32)  # shape = (input_dim, output_dim)

b = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # shape = (output_dim,)

# === 실행 ===
graph_executor.run_graph_cuda(
    E,
    len(E),
    shapes,
    len(shapes),
    W,
    b,
    W.shape[0],  # W_rows = input_dim
    W.shape[1]   # W_cols = output_dim
)
