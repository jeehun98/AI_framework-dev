import os, sys
import numpy as np

# === Import path & CUDA DLL 경로 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2 import _core as ge

np.random.seed(0)

# Unary
X = np.random.randn(2,3,4,5).astype(np.float32)
Y = ge.ewise_unary(X, kind="gelu")
# numpy ref(gelu: scipy 없음 → 근사 동일성은 대략성 확인)
def gelu_np(x):
    k0=0.7978845608; k1=0.044715
    return 0.5*x*(1+np.tanh(k0*(x+k1*x**3)))
ref = gelu_np(X)
print("Unary GELU close:", np.allclose(Y, ref, atol=1e-5))

# Binary broadcast
A = np.random.randn(2,3,1,1).astype(np.float32)
B = np.random.randn(1,3,4,5).astype(np.float32)
Z = ge.ewise_binary(A, B, kind="add")
print("Binary Add broadcast close:", np.allclose(Z, A + B, atol=1e-6))

# LeakyReLU(alpha)
Y2 = ge.ewise_unary(X, kind="leaky_relu", alpha=0.2)
ref2 = np.where(X>=0, X, 0.2*X)
print("Unary LeakyReLU close:", np.allclose(Y2, ref2, atol=1e-6))

# Div eps 보호
B2 = np.zeros((1,3,4,5), np.float32)
Z2 = ge.ewise_binary(X, B2, kind="div")
ref3 = X / np.maximum(B2, 1e-12)
print("Binary Div (eps) close:", np.allclose(Z2, ref3, atol=1e-6))
