import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.cal_graph.cal_graph import Cal_graph

def test_connect_graphs():
    cal_graph = Cal_graph()

    print("\n[Step 1] 첫 번째 계산 그래프 생성 (행렬 덧셈)")
    A = [[10, 10], [10, 10]]
    B = [[20, 20], [20, 20]]
    node_list1 = cal_graph.matrix_add(A, B)

    # ✅ 덧셈 연산의 리프 노드 찾기 (재귀적으로)
    leaf_node_list = cal_graph.get_leaf_nodes(node_list1)

    print("\n🔍 덧셈 연산의 리프 노드 리스트 출력:")
    for node in leaf_node_list:
        print(f"Leaf Node: {node.operation}")

    print("\n[Step 2] 두 번째 계산 그래프 생성 (행렬 곱)")
    C = [[1, 2], [3, 4]]
    D = [[5, 6], [7, 8]]
    node_list2 = cal_graph.matrix_multiply(C, D)

    print("\n[Step 3] 덧셈 그래프의 리프 노드에 행렬 곱 그래프 연결")
    cal_graph.connect_graphs(leaf_node_list, node_list2)
    cal_graph.print_graph()

    print(cal_graph.node_list, len(cal_graph.node_list), "test")

if __name__ == "__main__":
    test_connect_graphs()
