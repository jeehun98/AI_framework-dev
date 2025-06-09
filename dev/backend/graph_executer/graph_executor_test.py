import numpy as np
import os

# CUDA DLL 경로 명시 (Windows 전용 - Python 3.8 이상 필요)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

import graph_executor  # your compiled .pyd module

# E matrix (operation graph)
# Format: [op_type, input_id, param_id, output_id]
# op_type: 0=matmul, 1=add, 2=relu, 3=sigmoid
E = np.array([
    [0, 0, -1000, 1],   # matmul: buffer[0] @ param[0] -> buffer[1]
    [1, 1, -1001, 2],   # add:    buffer[1] + param[1] -> buffer[2]
    [3, 2, -1,    3],   # sigmoid: buffer[2] -> buffer[3]
    [0, 3, -1002, 4],   # matmul: buffer[3] @ param[2] -> buffer[4]
    [1, 4, -1003, 5],   # add:    buffer[4] + param[3] -> buffer[5]
    [3, 5, -1,    6],   # sigmoid: buffer[5] -> buffer[6]
], dtype=np.int32)

# Parameters: [W1, b1, W2, b2]
params = [
    np.random.randn(5, 4).astype(np.float32),  # W1: (5, 4)
    np.random.randn(1, 4).astype(np.float32),  # b1: (1, 4)
    np.random.randn(4, 3).astype(np.float32),  # W2: (4, 3)
    np.random.randn(1, 3).astype(np.float32),  # b2: (1, 3)
]

# Buffers: input + intermediate outputs
buffers = [
    np.random.randn(1, 5).astype(np.float32),  # buffer[0] = input
    np.zeros((1, 4), dtype=np.float32),        # buffer[1] = hidden matmul
    np.zeros((1, 4), dtype=np.float32),        # buffer[2] = add
    np.zeros((1, 4), dtype=np.float32),        # buffer[3] = sigmoid
    np.zeros((1, 3), dtype=np.float32),        # buffer[4] = matmul 2
    np.zeros((1, 3), dtype=np.float32),        # buffer[5] = add 2
    np.zeros((1, 3), dtype=np.float32),        # buffer[6] = output
]

# Check types before call
print("✅ Checking types...")
print("E:", type(E), E.dtype)
for i, p in enumerate(params):
    print(f"params[{i}]:", type(p), p.dtype, p.shape)
for i, b in enumerate(buffers):
    print(f"buffers[{i}]:", type(b), b.dtype, b.shape)

# Run the compiled CUDA-based execution
graph_executor.run_graph(E, params, buffers, input_id=0, output_id=6)

# Print output
print("\n✅ Final Output:")
print(buffers[6])
