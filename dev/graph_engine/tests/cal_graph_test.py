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

    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C, _ = matrix_ops.matrix_multiply(A_np, B_np)

    # ✅ 곱셈 그래프 생성 (root, leaf 반환)
    root1, leaf1 = cal_graph.add_matrix_multiply_graph(A, B, C.tolist())

    root1[0].print_tree()

    D = [[1, 2], [3, 4]]
    D_np = np.array(D, dtype=np.float64)
    E, _ = matrix_ops.matrix_add(C, D_np)

    print("덧셈 진입")

    # ✅ 덧셈 그래프 생성 (root, leaf 반환)
    root2, leaf2 = cal_graph.add_matrix_add_graph(C.tolist(), D, E.tolist())

    print(len(root1), len(leaf1), len(root2), len(leaf2))

    root2[0].print_tree()

    # ✅ 올바르게 root vs leaf 연결
    cal_graph.connect_graphs(root1, leaf2)

    print("???")

    cal_graph.print_graph()


if __name__ == "__main__":
    test_calculation_graph()
