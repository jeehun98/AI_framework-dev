import sys
import os
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import numpy as np
import graph_executor

# 공통 설정
batch = 2
input_dim = 3
output_dim = 4

# 가짜 E 행렬과 shapes 정보
E = np.array([], dtype=np.int32)  # 현재는 연산 순서를 안 씀
E_len = 0
shapes = np.array([1, batch, input_dim], dtype=np.int32)
shapes_len = len(shapes)

# Weight (3x4)와 Bias (1x4)
W = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
], dtype=np.float32)
b = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

W_rows, W_cols = W.shape

def run_test(activation_type, name):
    print(f"\n🔸 Activation: {name}")
    result = graph_executor.run_graph_cuda(
        E, E_len, shapes, shapes_len,
        W, b, W_rows, W_cols, activation_type
    )
    print("Result from GPU:")
    print(result)

# 활성화 함수별 테스트
run_test(0, "ReLU")
run_test(1, "Sigmoid")
run_test(2, "Tanh")
