# dev/backend/operaters/tests/test_matrix_ops.py

import os
import sys
import numpy as np

# ✅ 프로젝트 루트(dev/)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# ✅ 공통 경로 설정 및 CUDA 모듈 import
from tests.test_setup import setup_paths, import_cuda_module
setup_paths()
matrix_ops = import_cuda_module()

def test_matrix_mul():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[10, 20], [30, 40]], dtype=np.float64)

    result, _ = matrix_ops.matrix_multiply(A, B)

    expected = A @ B
    assert np.allclose(result, expected), f"Expected {expected.tolist()}, got {result.tolist()}"


if __name__ == "__main__":
    test_matrix_mul()
    print("✅ test_matrix_mul passed")
