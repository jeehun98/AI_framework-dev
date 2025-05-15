import sys
import os

# ✅ 프로젝트 루트 경로 추가

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
from dev.graph_engine.forward_from_graph import forward_from_graph
from dev.graph_engine.graph_compiler import GraphCompiler
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat

# 1️⃣ 레이어 구성 및 컴파일
layer1 = DenseMat(units=3, input_dim=4, initializer='he')
layer1.build(input_dim=4)
act1 = ActivationMat("sigmoid")
layer2 = DenseMat(units=2, initializer='he')
layer2.build(input_dim=3)
act2 = ActivationMat("sigmoid")

compiler = GraphCompiler()
compiler.output_ids = [0, 1, 2, 3]  # 입력 노드 ID (x[0], x[1], x[2], x[3])
compiler.node_offset = 4           # 다음 노드 인덱스 시작

compiler.add_layer(layer1)
compiler.add_layer(act1)
compiler.add_layer(layer2)
compiler.add_layer(act2)

graph = compiler.get_graph()

# 2️⃣ 입력값 지정
input_values = {
    0: 0.1,
    1: 0.4,
    2: 0.6,
    3: 0.8
}

# 3️⃣ 순전파 수행
Value = forward_from_graph(
    Conn=graph["Conn"],
    OpType=graph["OpType"],
    ParamIndex=graph["ParamIndex"],
    ParamValues=graph["ParamValues"],
    input_values=input_values
)



# 4️⃣ 최종 출력 노드 확인 및 출력
output_ids = graph["OutputIDs"]
print("📤 Final Outputs:")
for i in output_ids:
    print(f"Node {i}: {Value[i]}")

# 전체 그래프 내 주요 노드 값 확인
print("\n🧠 중간 노드 값 요약:")
for nid, val in enumerate(Value):
    if val is not None and nid >= 34:  # 첫 Dense 출력 이후부터
        print(f"Node {nid}: {val:.6f}")
