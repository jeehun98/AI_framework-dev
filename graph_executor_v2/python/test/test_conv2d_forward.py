# python/test/test_conv2d.py
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
def ge_conv2d_std(X, W_oi, B=None, *, sh=1, sw=1, ph=0, pw=0, dh=1, dw=1):
    """
    표준 계약(NCHW, OIHW, bias=(C_out,))을 ge.conv2d의 내부 계약에 맞춰 변환해 호출.
    - W: OIHW -> IOHW
    - B: (C_out,) -> (1,C_out,1,1)
    """
    W_in = np.transpose(W_oi, (1, 0, 2, 3))  # OIHW -> IOHW
    B_in = None if B is None else B.reshape(1, -1, 1, 1)
    return ge.conv2d(
        X, W_in, B_in,
        stride_h=sh, stride_w=sw,
        pad_h=ph, pad_w=pw,
        dil_h=dh, dil_w=dw
    )


# ---------- NumPy reference conv2d (NCHW) ----------
def conv2d_ref(X, W, B=None, stride=(1,1), pad=(0,0), dil=(1,1)):
    """
    X: (N, C_in, H, W)
    W: (C_out, C_in, KH, KW)
    B: (C_out,) or None
    """
    N, C_in, H, W_in = X.shape
    C_out, _, KH, KW = W.shape
    sh, sw = stride
    ph, pw = pad
    dh, dw = dil

    # 유효 커널 사이즈(팽창)
    eff_KH = (KH - 1) * dh + 1
    eff_KW = (KW - 1) * dw + 1

    H_out = (H + 2*ph - eff_KH) // sh + 1
    W_out = (W_in + 2*pw - eff_KW) // sw + 1

    Y = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    # 패딩 적용
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


# ---------- 수치미분 헬퍼 ----------
def numerical_grad(fn, x, eps=1e-3):
    """
    fn: returns scalar loss, takes x ndarray (view)
    x : ndarray to perturb (modified in-place)
    """
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f1 = fn()
        x[idx] = old - eps
        f2 = fn()
        x[idx] = old
        g[idx] = (f1 - f2) / (2*eps)
        it.iternext()
    return g


def allclose(name, a, b, atol=1e-4, rtol=1e-4):
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    print(f"{name}: {ok}")
    if not ok:
        diff = np.max(np.abs(a - b))
        print(f"  max abs diff = {diff}")
    return ok


def probe_kernel_mapping():
    N,Cin,Cout,H,W = 1,1,1,5,5
    KH,KW = 3,2
    X = np.zeros((N,Cin,H,W), np.float32)
    X[0,0,2,2] = 1.0  # 중심 근처 임펄스
    W = np.arange(KH*KW, dtype=np.float32).reshape(1,1,KH,KW)  # [[0,1],[2,3],[4,5]]
    B = np.array([0.0], np.float32)

    Y_ge = ge.conv2d(X, W, B, stride_h=1, stride_w=1, pad_h=0, pad_w=0, dil_h=1, dil_w=1)
    print("Y_ge[0,0]:\n", Y_ge[0,0])

    # 후보들: noflip / flip_h / flip_w / flip_hw / swap_hw ...
    def pp(name, Wvar):
        Y = conv2d_ref(X, Wvar, B, stride=(1,1), pad=(0,0), dil=(1,1))
        print(f"{name} diff:", float(np.max(np.abs(Y - Y_ge))))

    pp("ref_noflip", W)
    pp("flip_h", W[:, :, ::-1, :])
    pp("flip_w", W[:, :, :, ::-1])
    pp("flip_hw", W[:, :, ::-1, ::-1])
    Wsw = np.transpose(W, (0,1,3,2))
    pp("swap_hw", Wsw)
    pp("swap_hw+flip_h", Wsw[:, :, ::-1, :])
    pp("swap_hw+flip_w", Wsw[:, :, :, ::-1])
    pp("swap_hw+flip_hw", Wsw[:, :, ::-1, ::-1])

def test_forward_basic_nobias():
    N, C_in, H, W = 2, 3, 5, 4
    C_out, KH, KW = 4, 3, 2
    sh, sw = 2, 1
    ph, pw = 1, 1
    dh, dw = 1, 1

    X  = np.random.randn(N, C_in, H, W).astype(np.float32)       # NCHW
    Wt = np.random.randn(C_out, C_in, KH, KW).astype(np.float32) # OIHW

    Y_ref = conv2d_ref(X, Wt, None, stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
    Y_ge  = ge_conv2d_std(X, Wt, None, sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw)

    ok = allclose("forward no-bias (std-wrap)", Y_ge, Y_ref, atol=1e-4, rtol=1e-4)
    assert ok, "Forward(no-bias) 여전히 불일치하면 pad/stride H/W 스왑 이슈 가능"

def test_bias_broadcast():
    N, C_in, H, W = 1, 2, 3, 4
    C_out, KH, KW = 3, 1, 1
    X  = np.zeros((N, C_in, H, W), np.float32)
    Wt = np.zeros((C_out, C_in, KH, KW), np.float32)
    B  = np.arange(C_out, dtype=np.float32)  # [0,1,2]

    Y_ge = ge_conv2d_std(X, Wt, B, sh=1, sw=1, ph=0, pw=0, dh=1, dw=1)

    Y_expect = np.zeros((N, C_out, H, W), np.float32)
    for co in range(C_out):
        Y_expect[0, co, :, :] = B[co]

    ok = allclose("bias broadcast (std-wrap)", Y_ge, Y_expect, atol=1e-6, rtol=0)
    assert ok, "Bias 브로드캐스트 불일치 시 내부 계약이 다름"

def test_channel_mixing():
    N, C_in, H, W = 1, 2, 3, 3
    C_out, KH, KW = 2, 1, 1

    X = np.zeros((N, C_in, H, W), np.float32)
    X[0, 0, 1, 1] = 1.0
    X[0, 1, 1, 1] = 2.0

    Wt = np.zeros((C_out, C_in, KH, KW), np.float32)
    Wt[0,0,0,0] = 10.0; Wt[0,1,0,0] = 20.0
    Wt[1,0,0,0] = 30.0; Wt[1,1,0,0] = 40.0

    Y_ref = np.zeros((N, C_out, H, W), np.float32)
    Y_ref[0,0,1,1] = 50.0   # 1*10 + 2*20
    Y_ref[0,1,1,1] = 110.0  # 1*30 + 2*40

    Y_ge = ge_conv2d_std(X, Wt, None, sh=1, sw=1, ph=0, pw=0, dh=1, dw=1)
    ok = allclose("channel mixing (std-wrap)", Y_ge, Y_ref, atol=1e-6, rtol=0)
    assert ok, "채널 축 해석/가중치 레이아웃 변환이 아직 맞지 않음"

def test_forward_basic():
    N, C_in, H, W = 2, 3, 5, 4
    C_out, KH, KW = 4, 3, 2
    sh, sw = 2, 1
    ph, pw = 1, 1
    dh, dw = 1, 1

    X  = np.random.randn(N, C_in, H, W).astype(np.float32)
    Wt = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)
    Bt = np.random.randn(C_out).astype(np.float32)

    Y_ref = conv2d_ref(X, Wt, Bt, stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
    Y_ge  = ge_conv2d_std(X, Wt, Bt, sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw)

    ok = allclose("forward basic (std-wrap)", Y_ge, Y_ref, atol=1e-4, rtol=1e-4)
    assert ok, "forward basic mismatch (std-wrap)"


if __name__ == "__main__":
    
    # test_forward_basic()
    # test_forward_basic_nobias()
    #test_bias_broadcast()
    test_channel_mixing()
    
