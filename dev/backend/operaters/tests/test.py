# dev/backend/operaters/tests/inspect_operations_matrix_cuda.py

import os
import sys
import numpy as np

# ✅ 프로젝트 루트(dev/) 경로만 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# ✅ 공통 경로 설정 및 모듈 import
from tests.test_setup import setup_paths, import_cuda_module

setup_paths()
operations_matrix_cuda = import_cuda_module()


def print_node_list(node_list, title="Node list"):
    print(f"\n📦 {title}:")
    for i, node in enumerate(node_list):
        try:
            print(
                f"🧱 Node {i} → "
                f"Operation: {node.operation}, "
                f"Input: {node.input_value}, "
                f"Weight: {node.weight_value}, "
                f"Output: {node.output}, "
                f"Bias: {node.bias}"
            )
        except AttributeError as e:
            print(f"⚠️ AttributeError accessing node {i}: {e}")


# ✅ 테스트용 입력 데이터
A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

print("📊 Input A:\n", A)
print("📊 Input B:\n", B)

# ✅ matrix_add 테스트
try:
    print("\n🧪 Testing matrix_add...")
    result_add, node_list_add = operations_matrix_cuda.matrix_add(A, B)
    print("✅ Result of matrix_add:\n", result_add)
    print_node_list(node_list_add, "Nodes from matrix_add")
except Exception as e:
    print(f"❌ Error during matrix_add: {e}")

# ✅ matrix_multiply 테스트
try:
    print("\n🧪 Testing matrix_multiply...")
    result_mul, node_list_mul = operations_matrix_cuda.matrix_multiply(A, B)
    print("✅ Result of matrix_multiply:\n", result_mul)
    print_node_list(node_list_mul, "Nodes from matrix_multiply")
except Exception as e:
    print(f"❌ Error during matrix_multiply: {e}")
