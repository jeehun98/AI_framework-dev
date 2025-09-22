import numpy as np

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


from graph_executor_v2 import _core as ge

M,N = 4, 7
X = np.random.randn(M,N).astype(np.float32)
targets = np.random.randint(0, N, size=(M,), dtype=np.int64)

# forward
L = ge.cross_entropy(X, targets, reduction="mean")
# reference with numpy
x = X - X.max(axis=1, keepdims=True)
p = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
ref = (-np.log(p[np.arange(M), targets])).mean()
print("loss close:", np.allclose(L, ref, atol=1e-5))

# backward
dX = ge.cross_entropy_backward(X, targets, reduction="mean")
ref_dX = p.copy()
ref_dX[np.arange(M), targets] -= 1.0
ref_dX /= M
print("grad close:", np.allclose(dX, ref_dX, atol=1e-5))
