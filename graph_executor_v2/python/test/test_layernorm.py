import os, sys, numpy as np
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

def ln_ref(X, gamma=None, beta=None, eps=1e-5):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    y = (X - mu) / np.sqrt(var + eps)
    if gamma is not None: y = y * gamma
    if beta  is not None: y = y + beta
    return y

def test_forward():
    M, N = 8, 32
    X = np.random.randn(M,N).astype(np.float32)
    g = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    Y_ref = ln_ref(X, g, b)
    Y = core.layernorm(X, g, b)
    assert np.allclose(Y, Y_ref, atol=1e-5), np.abs(Y - Y_ref).max()

def test_backward_smoke():
    M, N = 4, 16
    X = np.random.randn(M,N).astype(np.float32)
    g = np.random.randn(N).astype(np.float32)
    Y = core.layernorm(X, g)
    dY = np.ones_like(Y, dtype=np.float32)
    dX, dgamma, dbeta = core.layernorm_backward(X, g, dY)
    # 기본 유효성
    assert np.isfinite(dX).all()
    assert np.isfinite(dgamma).all()
    assert np.isfinite(dbeta).all()

if __name__ == "__main__":
    test_forward()
    test_backward_smoke()
    print("LayerNorm ok")
