import numpy as np
import os
import sys

# ✅ 프로젝트 루트(dev/)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dev.layers.dense_mat import DenseMat

# 1️⃣ Dense 레이어 생성 및 초기화
layer = DenseMat(units=3, input_dim=4)
layer.build(input_dim=4)

# 2️⃣ 입력 노드 ID: 입력 4개를 0~3번 스칼라 노드로 가정
input_ids = [0, 1, 2, 3]
node_offset = 4  # 입력 이후부터 연산 노드 시작

# 3️⃣ 희소 그래프 조각 생성
block = layer.generate_sparse_matrix_block(input_ids, node_offset)

Conn = block["Conn"]
OpType = block["OpType"]
ParamIndex = block["ParamIndex"]
ParamValues = block["ParamValues"]
output_ids = block["output_ids"]

# 4️⃣ 결과 출력
print("🔗 Conn matrix (non-zero positions):")
conn_nonzero = np.argwhere(Conn == 1)
for i, j in conn_nonzero:
    print(f"Conn[{i}, {j}] = 1")

print("\n⚙️ OpType summary:")
unique, counts = np.unique(OpType[OpType > 0], return_counts=True)
for op, count in zip(unique, counts):
    print(f"OpType {op}: {count} nodes")

print("\n🧩 ParamValues count:", len(ParamValues))
print("📤 Output Node IDs:", output_ids)
print("🔚 Next Node Offset:", block["next_node_offset"])