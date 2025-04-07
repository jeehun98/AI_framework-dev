import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dev.node.node import Node
from dev.cal_graph.activations import build_sigmoid_node, build_tanh_node, build_relu_node
from dev.cal_graph.core_graph import Cal_graph

# ---------- 1. 그래프 준비 ----------

graph = Cal_graph()

# (1) matrix_add 를 통해 node_list 준비 (여기선 임의의 입력)
A = [[1.0, 2.0],
     [3.0, 4.0]]

B = [[5.0, 6.0],
     [7.0, 8.0]]

result = [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

graph.add_matrix_add_graph(A, B, result)
print("=== [Step 1] matrix_add 로 구성된 계산 그래프 ===")
graph.print_graph()

# ---------- 2. sigmoid 계산 그래프 연결 ----------

new_node_list = []
for node in graph.node_list:
    sigmoid_node = build_sigmoid_node(node)
    new_node_list.append(sigmoid_node)

graph.node_list = new_node_list  # node_list 교체

print("\n=== [Step 2] Sigmoid 계산 그래프 연결 후 ===")
graph.print_graph()