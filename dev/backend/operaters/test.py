import os
import sys
import numpy as np

# 빌드된 `operations_matrix_cuda` 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    print(f"Build path does not exist: {build_path}")
    sys.exit(1)

# CUDA DLL 경로 명시적 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    print(f"CUDA path does not exist: {cuda_path}")
    sys.exit(1)

# CUDA 확장 모듈 불러오기
try:
    import operations_matrix_cuda
    print("`operations_matrix_cuda` module imported successfully.")
except ImportError as e:
    print(f"Failed to import `operations_matrix_cuda`: {e}")
    sys.exit(1)

# 입력 데이터 생성
A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

# 행렬 덧셈 테스트
try:
    print("\nTesting matrix_add...")
    result_add, node_list_add = operations_matrix_cuda.matrix_add(A, B)

    print("Result of matrix_add:")
    print(result_add)

    print("Node list (matrix_add):")
    for i, node in enumerate(node_list_add):
        try:
            print(
                f"Node {i}: Operation: {node['operation']}, Input: {node['input_value']}, "
                f"Weight: {node['weight_value']}, Output: {node['output']}, Bias: {node['bias']}"
            )
        except KeyError as e:
            print(f"KeyError accessing attributes of node {i}: {e}")
except Exception as e:
    print(f"Error during matrix_add: {e}")

# 행렬 곱셈 테스트
try:
    print("\nTesting matrix_multiply...")
    result_mul, node_list_mul = operations_matrix_cuda.matrix_multiply(A, B)

    print("Result of matrix_multiply:")
    print(result_mul)

    print("Node list (matrix_multiply):")
    for i, node in enumerate(node_list_mul):
        try:
            print(
                f"Node {i}: Operation: {node['operation']}, Input: {node['input_value']}, "
                f"Weight: {node['weight_value']}, Output: {node['output']}, Bias: {node['bias']}"
            )
        except KeyError as e:
            print(f"KeyError accessing attributes of node {i}: {e}")
except Exception as e:
    print(f"Error during matrix_multiply: {e}")
