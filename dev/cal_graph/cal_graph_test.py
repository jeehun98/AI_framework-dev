import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.cal_graph.cal_graph import Cal_graph

def test_calculation_graph():
    cal_graph = Cal_graph()

    print("\n[Step 1] 초기 행렬 곱 연산 수행")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    # ✅ CUDA 또는 연산 결과값 (예제: NumPy로 행렬 곱 계산)
    result1 = np.dot(A, B).tolist()

    nodes1 = cal_graph.matrix_multiply(A, B, result1)  # ✅ result 추가

    # 노드 리스트 업데이트 (사용자가 직접 연결 가능)
    cal_graph.print_graph()

    cal_graph2 = Cal_graph()

    print("\n[Step 2] 행렬 덧셈 수행")
    C = [[10, 10], [10, 10]]
    D = [[20, 20], [20, 20]]

    # ✅ CUDA 또는 연산 결과값 (예제: NumPy로 행렬 덧셈 계산)
    result2 = (np.array(C) + np.array(D)).tolist()

    nodes2 = cal_graph2.matrix_add(C, D, result2)  # ✅ result 추가

    cal_graph2.print_graph()

if __name__ == "__main__":
    test_calculation_graph()
