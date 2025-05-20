import numpy as np
import pandas as pd

# Define op type codes
OP_TYPES = {
    "const": 0,
    "multiply": 1,
    "add": 2,
}

# A (4x4) @ B (4x4) = C (4x4)
# We express this with (N x N) matrices only: Conn, OpType, ParamIndex

# Total nodes: 32 const (A,B) + 64 mul + 48 add = 144
N = 144
Conn = np.zeros((N, N), dtype=np.int8)
OpType = np.zeros((N,), dtype=np.int32)
ParamIndex = np.full((N,), -1, dtype=np.int32)  # Only const nodes will get param index

# Map of parameters
param_values = []

# 1. A and B constants: node 0~31
for i in range(32):
    OpType[i] = OP_TYPES["const"]
    ParamIndex[i] = len(param_values)
    param_values.append(np.random.rand(1).item())  # scalar value

# 2. Multiply nodes (32~95): A[i][k] * B[k][j]
# Each C[i][j] has 4 multiplications, so total 16x4 = 64
idx = 32
for i in range(4):
    for j in range(4):
        for k in range(4):
            a_idx = i * 4 + k       # A[i][k] → node 0~15
            b_idx = 16 + k * 4 + j  # B[k][j] → node 16~31
            Conn[a_idx, idx] = 1
            Conn[b_idx, idx] = 1
            OpType[idx] = OP_TYPES["multiply"]
            idx += 1

# 3. Add nodes (96~143): sum 4 muls per output
# Each sum takes 3 additions: (((a + b) + c) + d)
for block in range(16):  # for each C[i][j]
    base = 32 + block * 4  # 4 mul outputs per C[i][j]
    a1 = idx
    Conn[base, a1] = 1
    Conn[base + 1, a1] = 1
    OpType[a1] = OP_TYPES["add"]
    idx += 1

    a2 = idx
    Conn[a1, a2] = 1
    Conn[base + 2, a2] = 1
    OpType[a2] = OP_TYPES["add"]
    idx += 1

    a3 = idx
    Conn[a2, a3] = 1
    Conn[base + 3, a3] = 1
    OpType[a3] = OP_TYPES["add"]
    idx += 1

# Final output nodes = 128~143
output_node_ids = list(range(128, 144))

df_preview = pd.DataFrame({
    "OpType": OpType[:30],
    "ParamIndex": ParamIndex[:30]
}).T

print("=== OpType Vector & ParamIndex Preview ===")
print(df_preview)

print("\n=== Conn Matrix (partial 30x30) ===")
print(pd.DataFrame(Conn[:50, :50]))
