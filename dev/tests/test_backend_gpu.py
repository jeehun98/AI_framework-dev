# tests/test_backend_gpu.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from backend import CPUBackend
from backend import CUDABackend

def test_gpu_match():
    be_cpu = CPUBackend()
    be_gpu = CUDABackend()

    A = np.random.randn(128, 64).astype(np.float32)
    B = np.random.randn(64, 32).astype(np.float32)
    x = np.random.randn(64).astype(np.float32)

    A_d = be_gpu.to_device(A)
    B_d = be_gpu.to_device(B)
    x_d = be_gpu.to_device(x)

    C_cpu = be_cpu.gemm(A, B)
    C_gpu = be_gpu.to_host(be_gpu.gemm(A_d, B_d))
    assert np.allclose(C_cpu, C_gpu, atol=1e-5)

    y_cpu = be_cpu.gemv(A, x)
    y_gpu = be_gpu.to_host(be_gpu.gemv(A_d, x_d))
    assert np.allclose(y_cpu, y_gpu, atol=1e-5)

if __name__ == "__main__":
    test_gpu_match()
    print("GPU backend OK")
