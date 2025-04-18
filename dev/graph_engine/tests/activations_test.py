import sys
import os

# ✅ 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dev.graph_engine.activations_graph import build_sigmoid_node, build_tanh_node, build_relu_node
from dev.graph_engine.node import Node

def test_activation_graph_structures():
     sigmoid_root = build_sigmoid_node()
     sigmoid_root.print_tree()

     sigmoid_root = build_sigmoid_node()
     print("children of reciprocal:", [c.operation for c in sigmoid_root.children])
     print("parents of reciprocal:", [p.operation for p in sigmoid_root.parents])

     # 부모도 자식도 트리 탐색용으로 확인
     for parent in sigmoid_root.parents:
          print(f"→ parent {parent.operation} has children: {[c.operation for c in parent.children]}")


if __name__ == "__main__":
    test_activation_graph_structures()

