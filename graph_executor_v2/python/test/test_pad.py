# python/test/test_pool2d.py
import os, sys
import numpy as np, torch
import torch.nn.functional as F

# === Import path & CUDA DLL 경로(Windows) ===
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
x = np.random.randn(2,3,4,5).astype(np.float32)
pads = [(0,0),(0,0),(1,2),(3,4)]  # (N,C,H,W) 기준
y = ge.pad(x, pads, value=-1.0)
yt = F.pad(torch.tensor(x), (3,4,1,2), value=-1.0)  # torch는 마지막 차원부터
print("FWD close:", np.allclose(y, yt.numpy(), atol=1e-6))

dy = np.ones_like(y, np.float32)
dx = ge.pad_backward(dy, pads, list(x.shape))
print("BWD close:", np.allclose(dx, np.ones_like(x), atol=1e-6))
