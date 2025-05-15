import sys
import os

# ✅ 프로젝트 루트 경로 추가

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat
from dev.graph_engine.graph_compiler import GraphCompiler

# 1️⃣ 레이어 구성: Dense → Sigmoid → Dense
layer1 = DenseMat(units=3, input_dim=4)
layer1.build(input_dim=4)
act1 = ActivationMat("sigmoid")
layer2 = DenseMat(units=2)
layer2.build(input_dim=3)
act2 = ActivationMat("sigmoid")

# 2️⃣ 컴파일러에 레이어 추가
compiler = GraphCompiler()
compiler.output_ids = [0, 1, 2, 3]  # 초기 입력 노드 ID들
compiler.node_offset = 4           # 입력 이후부터 노드 시작

compiler.add_layer(layer1)
compiler.add_layer(act1)
compiler.add_layer(layer2)
compiler.add_layer(act2)

# 3️⃣ 결과 확인
graph = compiler.get_graph()
Conn = graph["Conn"]
OpType = graph["OpType"]
ParamIndex = graph["ParamIndex"]
ParamValues = graph["ParamValues"]
OutputIDs = graph["OutputIDs"]
TotalNodes = graph["TotalNodes"]

print("🔗 Conn (non-zero entries):")
for i, j in np.argwhere(Conn == 1):
    print(f"Conn[{i}, {j}] = 1")

print("\n⚙️ OpType summary:")
unique, counts = np.unique(OpType[OpType > 0], return_counts=True)
for op, count in zip(unique, counts):
    print(f"OpType {op}: {count} nodes")

print("\n📦 Param count:", len(ParamValues))
print("📤 Output Node IDs:", OutputIDs)
print("🔚 Total Node Count:", TotalNodes)
