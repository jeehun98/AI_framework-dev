import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from dev.graph_engine.core_graph import Cal_graph

def test_connect_graphs():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]

    # ✅ 행렬 곱
    result1 = np.dot(A, B).tolist()
    node_list1 = cal_graph.add_matrix_multiply_graph(A, B, result1)

    C = [[1, 2], [3, 4]]
    result2 = (np.array(result1) + np.array(C)).tolist()
    node_list2 = cal_graph.add_matrix_add_graph(result1, C, result2)

    # ✅ 그래프 연결 (곱셈 → 덧셈)
    cal_graph.connect_graphs(node_list2, node_list1)

    # ✅ 그래프 출력
    cal_graph.print_graph()

    print(len(node_list1), len(node_list2))


if __name__ == "__main__":
    test_connect_graphs()
