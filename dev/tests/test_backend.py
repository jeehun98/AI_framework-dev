# tests/test_backend.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from backend import CPUBackend

def test_cpu_backend():
    be = CPUBackend()
    A = np.random.randn(8, 5).astype(np.float64)
    B = np.random.randn(5, 3).astype(np.float64)
    x = np.random.randn(5).astype(np.float64)

    C = be.gemm(A, B)
    y = be.gemv(A, x)
    assert C.shape == (8, 3)
    assert y.shape == (8,)

    # ridge 해석해법 체크
    XtX = be.gemm(A, A, transA=True)
    Xty = be.gemv(A, np.random.randn(8), transA=True)
    w = be.cholesky_solve(XtX, Xty, alpha=0.1)
    assert w.shape == (A.shape[1],)

if __name__ == "__main__":
    test_cpu_backend()
    print("CPU backend OK")
