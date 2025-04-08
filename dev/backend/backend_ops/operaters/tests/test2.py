import os
import sys
import numpy as np

# dev 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# 공통 설정
from tests.test_setup import setup_paths, import_cuda_module

setup_paths()
matrix_ops = import_cuda_module()

# ✅ float64로 타입 통일
A = np.random.randint(0, 4, (4, 4)).astype(np.float64)
B = np.random.randint(0, 4, (4, 4)).astype(np.float64)
C = np.zeros((4, 4), dtype=np.float64)

# ✅ matrix_add 테스트
try:
    print("🧪 Testing CUDA matrix_add...")
    result_add, _ = matrix_ops.matrix_add(A, B)
    print("✅ CUDA Addition Result:\n", result_add)
    assert np.allclose(result_add, A + B), f"❌ Addition mismatch!\nExpected:\n{A + B}\nGot:\n{result_add}"
    print("✅ Addition test passed!\n")
except Exception as e:
    print(f"❌ Error in CUDA addition: {e}")

# ✅ matrix_multiply 테스트
try:
    print("🧪 Testing CUDA matrix_multiply...")
    result_mul, _ = matrix_ops.matrix_multiply(A, B)
    print("✅ CUDA Multiplication Result:\n", result_mul)
    expected = np.dot(A, B)
    assert np.allclose(result_mul, expected), f"❌ Multiplication mismatch!\nExpected:\n{expected}\nGot:\n{result_mul}"
    print("✅ Multiplication test passed!\n")
except Exception as e:
    print(f"❌ Error in CUDA multiplication: {e}")
