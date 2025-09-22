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


from graph_executor_v2 import _core as ge

M,N=4,7
X=np.random.randn(M,N).astype(np.float32)
Y = ge.softmax(X)            # mask 없음
dY= np.random.randn(M,N).astype(np.float32)
dX = ge.softmax_backward(Y, dY)

# 참조 비교
ref = np.exp(X - X.max(1,keepdims=True)); 
ref = ref / ref.sum(1,keepdims=True)
ref_dX = (dY - (dY*ref).sum(1,keepdims=True)) * ref

print(np.allclose(Y, ref, atol=1e-3))
print(np.allclose(dX, ref_dX, atol=1e-5))
