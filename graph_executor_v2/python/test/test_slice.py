import os, sys, numpy as np

# === Import path & DLL 경로 설정 ===
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# import path
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path: sys.path.insert(0, PKG)

from graph_executor_v2 import _core as ge

X = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
Y = ge.slice(X, start=[0,0,1], stop=[2,3,4], step=[1,1,2])
assert Y.shape == (2,3,2)
assert np.allclose(Y, X[:, :, 1:4:2])
print("SLICE OK")
