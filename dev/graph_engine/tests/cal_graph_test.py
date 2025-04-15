import numpy as np
import os
import sys

# ✅ 프로젝트 루트(dev/)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ✅ 공통 경로 및 환경 설정
from tests.test_setup import setup_paths, import_cuda_module
setup_paths()

# ✅ 모듈 이름 및 빌드 디렉토리 설정
module_name = "operations_matrix_cuda"  # 실제 .pyd 파일 이름에서 .pyd 제외한 부분
build_dir = os.path.abspath("dev/backend/backend_ops/operaters/build/lib.win-amd64-cpython-312")
matrix_ops = import_cuda_module(module_name, build_dir)

# ✅ 계산 그래프 모듈 import
from dev.graph_engine.core_graph import Cal_graph


def test_calculation_graph():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]

    # ✅ CUDA 백엔드 곱셈
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C, _ = matrix_ops.matrix_multiply(A_np, B_np)

    # ✅ 계산 그래프에 곱셈 그래프 추가
    node_list1 = cal_graph.add_matrix_multiply_graph(A, B, C.tolist())

    D = [[1, 2], [3, 4]]
    D_np = np.array(D, dtype=np.float64)
    E, _ = matrix_ops.matrix_add(C, D_np)

    # ✅ 계산 그래프에 덧셈 그래프 추가
    node_list2 = cal_graph.add_matrix_add_graph(C.tolist(), D, E.tolist())

    # ✅ 연결
    cal_graph.connect_graphs(node_list2, node_list1)

    # ✅ 그래프 출력
    cal_graph.print_graph()


if __name__ == "__main__":
    test_calculation_graph()
