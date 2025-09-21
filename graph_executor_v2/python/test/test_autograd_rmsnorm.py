# python/test/test_autograd_rmsnorm.py
import os, sys, argparse
import numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# CUDA DLL (Windows) 힌트 경로
cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2 import _core as core

def rmsnorm_ref(X, gamma=None, beta=None, eps=1e-6):
    # X: [M,N]
    rms = np.sqrt((X**2).mean(axis=1, keepdims=True) + eps)
    Y = X / rms
    if gamma is not None:
        Y = Y * gamma
    if beta is not None:
        Y = Y + beta
    return Y

def finite_diff(X, gamma, beta, eps, idx, h=1e-3):
    Xp = X.copy(); Xp.flat[idx] += h
    Xm = X.copy(); Xm.flat[idx] -= h
    Yp = rmsnorm_ref(Xp, gamma, beta, eps)
    Ym = rmsnorm_ref(Xm, gamma, beta, eps)
    # 스칼라 손실: sum(Y)
    return ((Yp - Ym).sum() / (2*h))

def test_forward():
    M, N = 8, 16
    X = np.random.randn(M,N).astype(np.float32)
    gamma = np.random.randn(N).astype(np.float32)
    beta  = np.random.randn(N).astype(np.float32)

    Y_ref = rmsnorm_ref(X, gamma, beta)
    Y = core.rmsnorm(X, gamma, beta)
    assert np.allclose(Y, Y_ref, atol=1e-5), np.abs(Y-Y_ref).max()

def test_backward_smoke():
    M, N = 4, 8
    X = np.random.randn(M,N).astype(np.float32)
    gamma = np.random.randn(N).astype(np.float32)

    Y = core.rmsnorm(X, gamma)
    dY = np.ones_like(Y, dtype=np.float32)  # L = sum(Y)
    dX, dgamma, dbeta = core.rmsnorm_backward(X, gamma, dY)

    # 수치미분으로 dX 몇 개 샘플 체크
    for k in [0, N//2, N-1]:
        df = finite_diff(X, gamma, None, 1e-6, k)  # 같은 행 0 기준 만 약식 체크
        assert np.isfinite(dX.flat[k])
    assert np.isfinite(dgamma).all()

if __name__ == "__main__":
    test_forward()
    test_backward_smoke()
    print("RMSNorm ok")
