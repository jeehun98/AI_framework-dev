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

x = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)

y = ge.permute(x, [1,0,2])
assert y.shape == (3,2,4)
assert np.allclose(y, np.transpose(x, (1,0,2)))

z = ge.expand(x[0:1, :, 0:1], [2,3,4])  # [1,3,1] -> [2,3,4]
assert z.shape == (2,3,4)
assert np.allclose(z, np.broadcast_to(x[0:1,:,0:1], (2,3,4)))

c = ge.contiguous(y)  # no-op처럼 보이지만, 디바이스 라운드트립
assert np.allclose(c, y)
print("permute/expand/contiguous OK")
