# python/test/test_pool2d.py
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

np.random.seed(0)

# ----- Gather test -----
X = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4) # [2,3,4]
axis = 1
# index shape == output shape; non-axis dims must match X
index = np.array([[0,2,1],[1,1,0]], dtype=np.int32).reshape(2,3)  # shape [2,3] -> broadcast to [2,3,?]
# 우리가 정의한 규칙: Index, Y는 X와 동일 rank. inner=prod(after axis)
# 간단히 index를 [2,3,1]로 확장해 inner=4에서 broadcasting이 아닌 per-element index를 동일하게 쓰고자
index = index[..., None].repeat(4, axis=-1)  # [2,3,4]
Y = ge.gather(X, index, axis=1)

# numpy 참조 (take_along_axis)
ref = np.take_along_axis(X, index, axis=1)
print("Gather close:", np.allclose(Y, ref))

# ----- ScatterAdd test -----
K = 5
Out = np.zeros((2,K,4), dtype=np.float32)
# 같은 Index/shape로 Src를 더한다 (axis=1, M=3)
Src = np.random.randn(2,3,4).astype(np.float32)
Index = np.array([[0,2,1],[4,1,0]], dtype=np.int32)[...,None].repeat(4,axis=-1)  # [2,3,4]
Out2 = ge.scatter_add(Out, Index, Src, axis=1)

# numpy ref
ref2 = Out.copy()
for o in range(2):
  for m in range(3):
    idx = Index[o,m,0]
    ref2[o, idx, :] += Src[o,m,:]
print("ScatterAdd close:", np.allclose(Out2, ref2, atol=1e-6))
