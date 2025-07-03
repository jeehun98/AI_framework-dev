import sys
import os
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import cupy as cp
import numpy as np
from graph_executor import OpStruct, Shape, run_graph_cuda

# 💡 CuPy GPU 배열 초기화
x = cp.array([[1.0, 2.0]], dtype=cp.float32)
W = cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32)
b = cp.array([[0.5, -0.5]], dtype=cp.float32)

print("✅ CuPy 데이터 확인:")
print("x:", x)
print("W:", W)
print("b:", b)

# 🧠 CUDA에 넘길 포인터 구성
tensors = {
    "x0": int(x.data.ptr),
    "W": int(W.data.ptr),
    "b": int(b.data.ptr),
}

E = [
    OpStruct(0, "x0", "W", "linear"),   # MATMUL
    OpStruct(1, "linear", "b", "out"),  # ADD
    OpStruct(3, "out", "", "act_out"),  # SIGMOID
]

shapes = {
    "x0": Shape(1, 2),
    "W": Shape(2, 2),
    "b": Shape(1, 2),
    "linear": Shape(1, 2),
    "out": Shape(1, 2),
    "act_out": Shape(1, 2),
}

out_host = np.zeros((1, 2), dtype=np.float32)
run_graph_cuda(E, tensors, shapes, out_host, final_output_id="act_out")

print("✅ 최종 출력 결과:")
print(out_host)
