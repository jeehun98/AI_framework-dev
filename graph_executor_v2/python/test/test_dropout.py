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
M,N=4,7
X = np.random.randn(M,N).astype(np.float32)

Y, Msk = ge.dropout(X, p=0.2, return_mask=True, seed=42)
# scale = 1/(1-p)=1.25
ref = (Msk.astype(np.float32) * X) * (1.0/(1.0-0.2))
print("fwd close:", np.allclose(Y, ref, atol=1e-6))

dY = np.random.randn(M,N).astype(np.float32)
dX = ge.dropout_backward(dY, Msk, p=0.2)
ref_dx = (Msk.astype(np.float32) * dY) * (1.0/(1.0-0.2))
print("bwd close:", np.allclose(dX, ref_dx, atol=1e-6))
