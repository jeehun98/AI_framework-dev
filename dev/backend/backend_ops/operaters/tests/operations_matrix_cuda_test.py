import numpy as np
import os
import sys

# ✅ 프로젝트 루트(dev/)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# ✅ 공통 테스트 설정 임포트
from tests.test_setup import setup_paths, import_cuda_module

setup_paths()
operations_matrix_cuda = import_cuda_module()

# ✅ 올바른 경로로 import 수정
from graph_engine.core_graph import Cal_graph


def test_connect_graphs():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]

    result1, _ = operations_matrix_cuda.matrix_multiply(
        np.array(A, dtype=np.float64),
        np.array(B, dtype=np.float64)
    )
    result1_list = result1.tolist()
    node_list1 = cal_graph.add_matrix_multiply_graph(A, B, result1_list)

    C = [[1, 2], [3, 4]]
    result2, _ = operations_matrix_cuda.matrix_add(
        np.array(result1_list, dtype=np.float64),
        np.array(C, dtype=np.float64)
    )
    result2_list = result2.tolist()
    node_list2 = cal_graph.add_matrix_add_graph(result1_list, C, result2_list)

    cal_graph.connect_graphs(node_list2, node_list1)

    cal_graph.print_graph()

    print("result2_list =", result2_list)

    expected = np.array(A) @ np.array(B) + np.array(C)
    assert np.allclose(np.array(result2_list), expected), f"Expected {expected.tolist()}, got {result2_list}"


if __name__ == "__main__":
    test_connect_graphs()
