import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.cal_graph.cal_graph import Cal_graph

def test_connect_graphs():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]

    # ✅ 행렬 곱 수행
    result1 = np.dot(A, B).tolist()
    # ✅ 계산 그래프 생성
    node_list1 = cal_graph.matrix_multiply(A, B, result1)  

    C = [[1, 2], [3, 4]]

    # ✅ `result1`을 입력으로 사용하여 행렬 덧셈 수행
    result2 = (np.array(result1) + np.array(C)).tolist()
    node_list2 = cal_graph.matrix_add(result1, C, result2)  # ✅ 덧셈 연산 수행

    # ✅ 덧셈 연산의 리프 노드 찾기
    leaf_node_list2 = cal_graph.get_leaf_nodes(node_list2)

    cal_graph.connect_graphs(node_list2, node_list1)  # ✅ 곱셈 결과를 덧셈 입력으로 연결

    cal_graph.print_graph()

if __name__ == "__main__":
    test_connect_graphs()
