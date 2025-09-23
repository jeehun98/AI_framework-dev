# python/test/test_conv2d_forward_only.py
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

# ---------- NumPy reference conv2d (NCHW) ----------
def conv2d_ref(X, W, B=None, stride=(1,1), pad=(0,0), dil=(1,1)):
    N, C_in, H, W_in = X.shape
    C_out, _, KH, KW  = W.shape
    sh, sw = stride
    ph, pw = pad
    dh, dw = dil

    eff_KH = (KH - 1) * dh + 1
    eff_KW = (KW - 1) * dw + 1
    H_out = (H + 2*ph - eff_KH) // sh + 1
    W_out = (W_in + 2*pw - eff_KW) // sw + 1
    Y = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    Xp = np.pad(X, ((0,0),(0,0),(ph,ph),(pw,pw)), mode="constant")

    for n in range(N):
        for co in range(C_out):
            for ho in range(H_out):
                for wo in range(W_out):
                    h0 = ho*sh
                    w0 = wo*sw
                    acc = 0.0
                    for ci in range(C_in):
                        for kh in range(KH):
                            ih = h0 + kh*dh
                            for kw in range(KW):
                                iw = w0 + kw*dw
                                acc += Xp[n, ci, ih, iw] * W[co, ci, kh, kw]
                    if B is not None:
                        acc += B[co]
                    Y[n, co, ho, wo] = acc
    return Y

def print_diff_stats(tag, Y, Y_ref):
    diff = Y - Y_ref
    adiff = np.abs(diff)
    mx = adiff.max()
    idx = np.unravel_index(adiff.argmax(), adiff.shape)
    print(f"[{tag}] allclose: {np.allclose(Y, Y_ref, atol=1e-4, rtol=1e-4)}")
    print(f"  max abs diff = {mx} at idx {idx}  "
          f"(Y={Y[idx]}, Y_ref={Y_ref[idx]}, diff={diff[idx]})")
    # 채널별 평균/최대 차이
    ch_max = adiff.reshape(adiff.shape[0], adiff.shape[1], -1).max(axis=2)
    ch_mean= adiff.reshape(adiff.shape[0], adiff.shape[1], -1).mean(axis=2)
    print(f"  per-channel max (N,Cout):\n{ch_max}")
    print(f"  per-channel mean (N,Cout):\n{ch_mean}")

def run_case(name, N,Cin,H,W, Cout,KH,KW, sh,sw, ph,pw, dh,dw,
             zero_W=False, zero_B=False, one_hot_W=False):
    print(f"\n=== {name} ===")
    X  = np.random.randn(N, Cin, H, W).astype(np.float32)
    Wt = np.random.randn(Cout, Cin, KH, KW).astype(np.float32)
    Bt = np.random.randn(Cout).astype(np.float32)

    if zero_W: Wt[:] = 0
    if zero_B: Bt[:] = 0
    if one_hot_W:
        Wt[:] = 0
        # 간단한 one-hot: 첫 출력채널/첫 입력채널/커널 좌상단 = 1
        Wt[0, 0, 0, 0] = 1.0

    Y     = ge.conv2d(X, Wt, Bt, stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw, dil_h=dh, dil_w=dw)
    Y_ref = conv2d_ref(X, Wt, None if zero_B else Bt,
                       stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))

    print_diff_stats(name, Y, Y_ref)

def main():
    # 공통 파라미터 (원래 테스트와 동일)
    N,Cin,H,W      = 2, 3, 5, 4
    Cout,KH,KW     = 4, 3, 2
    sh,sw          = 2, 1
    ph,pw          = 1, 1
    dh,dw          = 1, 1

    # 개별 케이스
    run_case("A: random W,B (기본)", N,Cin,H,W, Cout,KH,KW, sh,sw, ph,pw, dh,dw)
    run_case("B: Bt=0, W=random (bias 끔)", N,Cin,H,W, Cout,KH,KW, sh,sw, ph,pw, dh,dw, zero_B=True)
    run_case("C: W=0, B=random (bias만)", N,Cin,H,W, Cout,KH,KW, sh,sw, ph,pw, dh,dw, zero_W=True)
    run_case("D: W one-hot (0,0,0,0)=1, Bt=0", N,Cin,H,W, Cout,KH,KW, sh,sw, ph,pw, dh,dw, one_hot_W=True, zero_B=True)

if __name__ == "__main__":
    main()
