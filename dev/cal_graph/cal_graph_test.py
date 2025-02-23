import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.cal_graph.cal_graph import Cal_graph

def test_calculation_graph():
    cal_graph = Cal_graph()

    print("\n[Step 1] 초기 행렬 곱 연산 수행")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    nodes1 = cal_graph.matrix_multiply(A, B)
    cal_graph.print_graph()

    print("\n[Step 2] 기존 노드 리스트를 활용한 행렬 덧셈 수행")
    C = [[10, 10], [10, 10]]
    D = [[20, 20], [20, 20]]
    nodes2 = cal_graph.matrix_add(C, D, node_list=nodes1)
    cal_graph.print_graph()

    print("\n[Step 3] 새로운 행렬 덧셈 수행 (기존 노드 없이)")
    nodes3 = cal_graph.matrix_add([[5, 5], [5, 5]], [[10, 10], [10, 10]])
    cal_graph.print_graph()

if __name__ == "__main__":
    test_calculation_graph()
