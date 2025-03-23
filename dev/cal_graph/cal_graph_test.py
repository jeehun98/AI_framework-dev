# backend 를 통한 행렬 연산과 계산 그래프 연결 확인

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# 빌드된 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise FileNotFoundError(f"Build path does not exist: {build_path}")

# CUDA DLL 경로 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    raise FileNotFoundError(f"CUDA path does not exist: {cuda_path}")

try:
    import matrix_ops
except ImportError as e:
    raise ImportError("Failed to import `matrix_ops` module. Ensure it is built and the path is correctly set.") from e


from dev.cal_graph.cal_graph import Cal_graph

def test_calculation_graph():
    cal_graph = Cal_graph()

    print("\n[Step 1] 초기 행렬 곱 연산 수행")

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]
    C = np.zeros((2, 2), dtype=np.float32)

    # ✅ CUDA 또는 연산 결과값 (예제: NumPy로 행렬 곱 계산)
    matrix_ops.matrix_mul(A, B, C)

    node_list1 = cal_graph.matrix_multiply(A, B, C)  # ✅ result 추가

    # 노드 리스트 업데이트 (사용자가 직접 연결 가능)
    D = [[1, 2], [3, 4]]

    E = np.zeros((2, 2), dtype=np.float32)
    # ✅ CUDA 또는 연산 결과값 (예제: NumPy로 행렬 덧셈 계산)
    result2 = matrix_ops.matrix_add(C, D, E)

    node_list2 = cal_graph.matrix_add(C, D, E)  # ✅ result 추가

    leaf_node_list = cal_graph.get_leaf_nodes(node_list2)

    cal_graph.connect_graphs(node_list2, node_list1)

    cal_graph.print_graph()

if __name__ == "__main__":
    test_calculation_graph()
