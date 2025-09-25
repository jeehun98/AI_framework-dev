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

np.random.seed(1)
N,C,H,W = 1,2,3,4
X = np.random.randn(N,C,H,W).astype(np.float32)

# forward
Y = ge.upsample2d_bilinear(X, out_h=None, out_w=None, scale_h=2.0, scale_w=2.0, align_corners=False)
Ho, Wo = Y.shape[2], Y.shape[3]

# naive ref (align_corners=False)
ref = np.empty_like(Y)
scale_h = H / Ho
scale_w = W / Wo
for n in range(N):
  for c in range(C):
    for ho in range(Ho):
      src_h = (ho + 0.5)*scale_h - 0.5
      h0 = int(np.floor(src_h)); h1 = h0 + 1
      ah = src_h - h0; bh = 1 - ah
      h0 = max(0, min(H-1, h0)); h1 = max(0, min(H-1, h1))
      for wo in range(Wo):
        src_w = (wo + 0.5)*scale_w - 0.5
        w0 = int(np.floor(src_w)); w1 = w0 + 1
        aw = src_w - w0; bw = 1 - aw
        w0 = max(0, min(W-1, w0)); w1 = max(0, min(W-1, w1))
        v00 = X[n,c,h0,w0]; v01 = X[n,c,h0,w1]
        v10 = X[n,c,h1,w0]; v11 = X[n,c,h1,w1]
        top = bw*v00 + aw*v01
        bottom = bw*v10 + aw*v11
        ref[n,c,ho,wo] = bh*top + ah*bottom

print("Upsample2D(bilinear) FWD close:", np.allclose(Y, ref, atol=1e-5))

# backward (dY=ones)
dY = np.ones_like(Y, dtype=np.float32)
dX = ge.upsample2d_bilinear_backward(dY, H, W, align_corners=False)

# ref grad accumulation
ref_dX = np.zeros_like(X)
for n in range(N):
  for c in range(C):
    for ho in range(Ho):
      src_h = (ho + 0.5)*scale_h - 0.5
      h0 = int(np.floor(src_h)); h1 = h0 + 1
      ah = src_h - h0; bh = 1 - ah
      h0 = max(0, min(H-1, h0)); h1 = max(0, min(H-1, h1))
      for wo in range(Wo):
        src_w = (wo + 0.5)*scale_w - 0.5
        w0 = int(np.floor(src_w)); w1 = w0 + 1
        aw = src_w - w0; bw = 1 - aw
        w0 = max(0, min(W-1, w0)); w1 = max(0, min(W-1, w1))
        g = 1.0
        ref_dX[n,c,h0,w0] += g*(bh*bw)
        ref_dX[n,c,h0,w1] += g*(bh*aw)
        ref_dX[n,c,h1,w0] += g*(ah*bw)
        ref_dX[n,c,h1,w1] += g*(ah*aw)

print("Upsample2D(bilinear) BWD close:", np.allclose(dX, ref_dX, atol=1e-5))
