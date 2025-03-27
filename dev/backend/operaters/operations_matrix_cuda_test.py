import os
import sys
import numpy as np

# ✅ 1. operations_matrix_cuda 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    print(f"Build path does not exist: {build_path}")
    sys.exit(1)

# ✅ 2. 프로젝트 루트 경로 추가 (dev가 포함된 경로)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ 3. CUDA DLL 경로 명시적 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    print(f"CUDA path does not exist: {cuda_path}")
    sys.exit(1)

# ✅ 4. CUDA 연산 모듈 임포트
try:
    import operations_matrix_cuda
    print("`operations_matrix_cuda` module imported successfully.")
except ImportError as e:
    print(f"Failed to import `operations_matrix_cuda`: {e}")
    sys.exit(1)

# ✅ 5. 계산 그래프 Cal_graph 임포트
from dev.cal_graph.cal_graph import Cal_graph


def test_connect_graphs():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]

    # ✅ CUDA 행렬 곱 수행
    result1, _ = operations_matrix_cuda.matrix_multiply(np.array(A, dtype=np.float64),
                                                        np.array(B, dtype=np.float64))
    result1_list = result1.tolist()

    # ✅ 계산 그래프 생성
    node_list1 = cal_graph.matrix_multiply(A, B, result1_list)

    C = [[1, 2], [3, 4]]

    # ✅ CUDA 행렬 덧셈 수행
    result2, _ = operations_matrix_cuda.matrix_add(np.array(result1_list, dtype=np.float64),
                                                np.array(C, dtype=np.float64))
    result2_list = result2.tolist()

    # ✅ 덧셈 노드 생성
    node_list2 = cal_graph.matrix_add(result1_list, C, result2_list)

    # ✅ 그래프 연결
    leaf_node_list2 = cal_graph.get_leaf_nodes(node_list2)
    cal_graph.connect_graphs(node_list2, node_list1)

    cal_graph.print_graph()


if __name__ == "__main__":
    test_connect_graphs()
