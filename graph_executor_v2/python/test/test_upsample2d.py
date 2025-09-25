# python/test/test_upsample2d.py
import os, sys, numpy as np
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

np.random.seed(0)
N,C,H,W = 2,3,4,5
X = np.random.randn(N,C,H,W).astype(np.float32)

# ----- FWD: scale -----
Y = ge.upsample2d_nearest(X, out_h=None, out_w=None, scale_h=2.0, scale_w=3.0, align_corners=False)
Ho, Wo = Y.shape[2], Y.shape[3]

# NumPy ref (nearest, align_corners=False)
ref = np.empty_like(Y)
scale_h = H / Ho
scale_w = W / Wo
for n in range(N):
  for c in range(C):
    for ho in range(Ho):
      ih = int(np.floor((ho + 0.5)*scale_h - 0.5))
      ih = max(0, min(H-1, ih))
      for wo in range(Wo):
        iw = int(np.floor((wo + 0.5)*scale_w - 0.5))
        iw = max(0, min(W-1, iw))
        ref[n,c,ho,wo] = X[n,c,ih,iw]

print("Upsample2D(nearest) FWD close:", np.allclose(Y, ref, atol=1e-6))

# ----- BWD -----
dY = np.ones_like(Y, dtype=np.float32)
dX = ge.upsample2d_nearest_backward(dY, H, W, align_corners=False)

# ref grad: 각 원소가 몇 번 복제되었는지 카운트
ref_dX = np.zeros_like(X)
for n in range(N):
  for c in range(C):
    for ho in range(Ho):
      ih = int(np.floor((ho + 0.5)*scale_h - 0.5))
      ih = max(0, min(H-1, ih))
      for wo in range(Wo):
        iw = int(np.floor((wo + 0.5)*scale_w - 0.5))
        iw = max(0, min(W-1, iw))
        ref_dX[n,c,ih,iw] += 1.0

print("Upsample2D(nearest) BWD close:", np.allclose(dX, ref_dX, atol=1e-6))
