import sys
import os

# ✅ 프로젝트 루트 경로 추가

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from dev.graph_engine.graph_compiler import GraphCompiler
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat

# ✅ 레이어 구성
layer1 = DenseMat(units=3, input_dim=4)
layer1.build(input_dim=4)
act1 = ActivationMat("sigmoid")
layer2 = DenseMat(units=2)
layer2.build(input_dim=3)
act2 = ActivationMat("sigmoid")

# ✅ 컴파일러 생성 및 레이어 추가
compiler = GraphCompiler()
compiler.output_ids = [0, 1, 2, 3]
compiler.node_offset = 4

compiler.add_layer(layer1)
compiler.add_layer(act1)
compiler.add_layer(layer2)
compiler.add_layer(act2)

graph = compiler.get_graph()
optype_map = graph["OpTypeNodeMap"]

# ✅ 결과 출력
print("\n📊 [OpType별 노드 분해 결과]")
for op, nodes in optype_map.items():
    print(f"OpType {op}: {len(nodes)} nodes → {nodes[:10]}{'...' if len(nodes) > 10 else ''}")
