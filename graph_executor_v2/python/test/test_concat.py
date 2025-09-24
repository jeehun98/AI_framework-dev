import os, sys
import numpy as np

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

def t_axis1():
    A = np.ones((2,3,4), np.float32)
    B = 2*np.ones((2,3,4), np.float32)
    C = ge.concat([A,B], axis=1)
    assert C.shape==(2,6,4)
    assert np.allclose(C[:,:3], 1) and np.allclose(C[:,3:], 2)
    print("axis1 ok")

def t_axis0():
    A = np.full((2,3,4), 3, np.float32)
    B = np.full((1,3,4), 4, np.float32)
    C = ge.concat([A,B], axis=0)
    assert C.shape==(3,3,4)
    assert np.allclose(C[:2], 3) and np.allclose(C[2:], 4)
    print("axis0 ok")

def t_axis2():
    A = np.full((2,3,2), 5, np.float32)
    B = np.full((2,3,1), 6, np.float32)
    C = ge.concat([A,B], axis=2)
    assert C.shape==(2,3,3)
    assert np.allclose(C[:,:,:2], 5) and np.allclose(C[:,:,2:], 6)
    print("axis2 ok")

t_axis1(); t_axis0(); t_axis2()
print("ALL CONCAT OK")
