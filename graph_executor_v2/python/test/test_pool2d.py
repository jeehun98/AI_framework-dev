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

# ---- 공용 출력 크기 계산 (런처/커널과 동일 공식) ----
def out_hw_2d(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False):
    effKH = (kH - 1) * dH + 1
    effKW = (kW - 1) * dW + 1
    aH = H + 2 * pH - effKH
    aW = W + 2 * pW - effKW
    if ceil_mode:
        Ho = (aH + sH - 1) // sH + 1 if aH >= 0 else 0
        Wo = (aW + sW - 1) // sW + 1 if aW >= 0 else 0
    else:
        Ho = (aH // sH) + 1 if aH >= 0 else 0
        Wo = (aW // sW) + 1 if aW >= 0 else 0
    Ho = max(Ho, 0)
    Wo = max(Wo, 0)
    return Ho, Wo

np.random.seed(0)

# ------------ 테스트 파라미터 ------------
N, C, H, W = 2, 3, 5, 6
kH, kW = 2, 2
sH, sW = 2, 2
pH, pW = 0, 0
dH, dW = 1, 1
ceil_mode = False
count_include_pad = False

X = np.random.randn(N, C, H, W).astype(np.float32)

# ============ MaxPool2D ============
Y, Ind = ge.maxpool2d(
    X,
    kH=kH, kW=kW,
    sH=sH, sW=sW,
    pH=pH, pW=pW,
    dH=dH, dW=dW,
    ceil_mode=ceil_mode,
    return_indices=True,
)

Ho, Wo = out_hw_2d(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)

# 모양 점검 (디버그 메시지 도움)
if Y.shape != (N, C, Ho, Wo):
    print("[WARN] CUDA Y.shape:", Y.shape, " expected:", (N, C, Ho, Wo))
if Ind.shape != (N, C, Ho, Wo):
    print("[WARN] CUDA Ind.shape:", Ind.shape, " expected:", (N, C, Ho, Wo))

# NumPy ref (Max)
ref = np.empty((N, C, Ho, Wo), np.float32)
for n in range(N):
    for c in range(C):
        for ho in range(Ho):
            for wo in range(Wo):
                hs, ws = ho * sH - pH, wo * sW - pW
                vals = []
                for kh in range(kH):
                    ih = hs + kh * dH
                    if ih < 0 or ih >= H:
                        continue
                    for kw_ in range(kW):
                        iw = ws + kw_ * dW
                        if iw < 0 or iw >= W:
                            continue
                        vals.append(X[n, c, ih, iw])
                ref[n, c, ho, wo] = np.max(vals) if len(vals) > 0 else -np.inf

print("MaxPool2D FWD close:", np.allclose(Y, ref, atol=1e-6))

# backward: dY = ones → dX should place ones at argmax locations
dY = np.ones_like(Y, dtype=np.float32)
dX = ge.maxpool2d_backward(
    dY, Ind, H, W,
    kH=kH, kW=kW,
    sH=sH, sW=sW,
    pH=pH, pW=pW,
    dH=dH, dW=dW,
    ceil_mode=ceil_mode
)

# reference bwd
ref_dX = np.zeros_like(X)
for n in range(N):
    for c in range(C):
        for ho in range(Ho):
            for wo in range(Wo):
                idx = int(Ind[n, c, ho, wo])
                ih, iw = idx // W, idx % W
                if 0 <= ih < H and 0 <= iw < W:
                    ref_dX[n, c, ih, iw] += 1.0
print("MaxPool2D BWD close:", np.allclose(dX, ref_dX, atol=1e-6))

# ============ AvgPool2D ============
Y2 = ge.avgpool2d(
    X,
    kH=kH, kW=kW,
    sH=sH, sW=sW,
    pH=pH, pW=pW,
    dH=dH, dW=dW,
    ceil_mode=ceil_mode,
    count_include_pad=count_include_pad
)

# NumPy ref (Avg)
ref2 = np.empty((N, C, Ho, Wo), np.float32)
for n in range(N):
    for c in range(C):
        for ho in range(Ho):
            for wo in range(Wo):
                hs, ws = ho * sH - pH, wo * sW - pW
                ssum = 0.0
                cnt = 0
                for kh in range(kH):
                    ih = hs + kh * dH
                    for kw_ in range(kW):
                        iw = ws + kw_ * dW
                        if 0 <= ih < H and 0 <= iw < W:
                            ssum += X[n, c, ih, iw]
                            cnt += 1
                        elif count_include_pad:
                            cnt += 1
                if cnt == 0:
                    cnt = 1
                ref2[n, c, ho, wo] = ssum / float(cnt)

print("AvgPool2D FWD close:", np.allclose(Y2, ref2, atol=1e-6))

# Avg backward
dY2 = np.ones_like(Y2, dtype=np.float32)
dX2 = ge.avgpool2d_backward(
    dY2, H, W,
    kH=kH, kW=kW,
    sH=sH, sW=sW,
    pH=pH, pW=pW,
    dH=dH, dW=dW,
    ceil_mode=ceil_mode,
    count_include_pad=count_include_pad
)

ref_dX2 = np.zeros_like(X)
for n in range(N):
    for c in range(C):
        for ho in range(Ho):
            for wo in range(Wo):
                hs, ws = ho * sH - pH, wo * sW - pW
                # 분모(cnt) 계산 (FWD 규칙과 동일)
                cnt = 0
                for kh in range(kH):
                    ih = hs + kh * dH
                    for kw_ in range(kW):
                        iw = ws + kw_ * dW
                        if 0 <= ih < H and 0 <= iw < W:
                            cnt += 1
                        elif count_include_pad:
                            cnt += 1
                if cnt == 0:
                    cnt = 1
                g = 1.0 / float(cnt)  # dY2=1

                # 기여 분배
                for kh in range(kH):
                    ih = hs + kh * dH
                    if ih < 0 or ih >= H:
                        continue
                    for kw_ in range(kW):
                        iw = ws + kw_ * dW
                        if iw < 0 or iw >= W:
                            # count_include_pad 는 입력 메모리 없음 → 버림
                            continue
                        ref_dX2[n, c, ih, iw] += g

print("AvgPool2D BWD close:", np.allclose(dX2, ref_dX2, atol=1e-6))
