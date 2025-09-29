# test_conv2d.py
import os, sys, argparse
import numpy as np

# === Import path & CUDA DLL 경로 (Windows) ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..",".."))
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

from graph_executor_v2.ops import require

# --- Try import CuPy (preferred for device allocations) ---
try:
    import cupy as cp
    HAS_CUPY = True
except Exception as e:
    HAS_CUPY = False
    cp = None

ops_conv2d = require("conv2d")  # -> _ops_conv2d
Conv2DAttrs = ops_conv2d.Conv2DAttrs

def ho_wo(H, W, Kh, Kw, sh, sw, ph, pw, dh, dw):
    Ho = (H + 2*ph - dh*(Kh-1) - 1)//sh + 1
    Wo = (W + 2*pw - dw*(Kw-1) - 1)//sw + 1
    return Ho, Wo

def conv2d_ref_nchw(x, w, b=None, stride=(1,1), pad=(0,0), dil=(1,1)):
    """
    Reference conv (NCHW, OIHW), float32. CPU numpy slow but OK for tiny tensors.
    stride=(sh,sw), pad=(ph,pw), dil=(dh,dw)
    """
    N, Cin, H, W = x.shape
    Cout, Cin2, Kh, Kw = w.shape
    assert Cin == Cin2
    sh, sw = stride
    ph, pw = pad
    dh, dw = dil
    Ho, Wo = ho_wo(H, W, Kh, Kw, sh, sw, ph, pw, dh, dw)
    y = np.zeros((N, Cout, Ho, Wo), dtype=np.float32)
    for n in range(N):
        for co in range(Cout):
            for ho in range(Ho):
                for wo in range(Wo):
                    acc = 0.0
                    for ci in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                h_in = ho*sh + kh*dh - ph
                                w_in = wo*sw + kw*dw - pw
                                if 0 <= h_in < H and 0 <= w_in < W:
                                    acc += x[n, ci, h_in, w_in] * w[co, ci, kh, kw]
                    if b is not None:
                        acc += b[co]
                    y[n, co, ho, wo] = acc
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite-diff", action="store_true",
                    help="작은 텐서에서 간단 참조 구현과 비교")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    print("LOADED:", ops_conv2d.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_conv2d expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # --- Shapes (tiny for correctness) ---
    N, Cin, H, W = 2, 3, 6, 5
    Cout, Kh, Kw = 4, 3, 3
    sh, sw = 1, 1
    ph, pw = 1, 1
    dh, dw = 1, 1
    Ho, Wo = ho_wo(H, W, Kh, Kw, sh, sw, ph, pw, dh, dw)

    # --- Host data ---
    x_h = rng.standard_normal(size=(N, Cin, H, W), dtype=np.float32)
    w_h = rng.standard_normal(size=(Cout, Cin, Kh, Kw), dtype=np.float32)
    b_h = rng.standard_normal(size=(Cout,), dtype=np.float32)

    # --- Device buffers (CuPy) ---
    x_d = cp.asarray(x_h)              # [N,Cin,H,W]
    w_d = cp.asarray(w_h)              # [Cout,Cin,Kh,Kw]
    y_d = cp.zeros((N, Cout, Ho, Wo), dtype=cp.float32)
    b_d = cp.asarray(b_h)

    # --- Attrs ---
    attrs = Conv2DAttrs()
    attrs.stride_h, attrs.stride_w = sh, sw
    attrs.pad_h, attrs.pad_w       = ph, pw
    attrs.dil_h, attrs.dil_w       = dh, dw
    attrs.groups                   = 1

    # --- Forward ---
    ops_conv2d.forward(
        int(x_d.data.ptr), [N, Cin, H, W],
        int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
        int(y_d.data.ptr), [N, Cout, Ho, Wo],
        int(b_d.data.ptr),  # bias_ptr (or None)
        attrs,
        0  # stream=0 (default)
    )
    y_h = cp.asnumpy(y_d)
    print("Y.shape:", y_h.shape)
    assert y_h.shape == (N, Cout, Ho, Wo)

    # --- Quick reference check (tiny tensors) ---
    if args.finite_diff:  # we also use this flag to run reference compare
        y_ref = conv2d_ref_nchw(x_h, w_h, b_h, stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
        max_abs = float(np.max(np.abs(y_h - y_ref)))
        print("Forward max_abs diff vs ref:", max_abs)
        # im2col+gemm + fp32 should be very close on tiny sizes
        assert max_abs < 5e-4, f"forward mismatch: max_abs={max_abs}"

    # --- Backward quick checks ---
    # We check dB correctness (sum over N, H, W) and shapes for dW/dX.
    gy_h = rng.standard_normal(size=(N, Cout, Ho, Wo), dtype=np.float32)
    gy_d = cp.asarray(gy_h)

    # dB only
    db_d = cp.zeros((Cout,), dtype=cp.float32)
    ops_conv2d.backward(
        int(x_d.data.ptr), [N, Cin, H, W],
        int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        None,                         # dW
        int(db_d.data.ptr),           # dB
        None,                         # dX
        attrs, 0
    )
    db_h = cp.asnumpy(db_d)
    # reference: sum over N,H,W of gy
    db_ref = gy_h.sum(axis=(0, 2, 3))
    max_abs_db = float(np.max(np.abs(db_h - db_ref)))
    print("dB max_abs diff:", max_abs_db)
    assert max_abs_db < 5e-4, f"dB mismatch: {max_abs_db}"

    # dW only (sanity: shape & basic finite diff on 1 element)
    dw_d = cp.zeros((Cout, Cin, Kh, Kw), dtype=cp.float32)
    ops_conv2d.backward(
        int(x_d.data.ptr), [N, Cin, H, W],
        int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        int(dw_d.data.ptr), # dW
        None,               # dB
        None,               # dX
        attrs, 0
    )
    dw_h = cp.asnumpy(dw_d)
    assert dw_h.shape == (Cout, Cin, Kh, Kw)

    # tiny finite diff for a single dW element (co,ci,kh,kw)=(0,0,0,0)
    # loss = <Y, gy>
    def forward_only(x, w, b):
        y = conv2d_ref_nchw(x, w, b, stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
        return float((y * gy_h).sum())

    if args.finite_diff:
        eps = 1e-3
        w_pos = w_h.copy(); w_pos[0,0,0,0] += eps
        w_neg = w_h.copy(); w_neg[0,0,0,0] -= eps
        loss_pos = forward_only(x_h, w_pos, b_h)
        loss_neg = forward_only(x_h, w_neg, b_h)
        d_fd = (loss_pos - loss_neg) / (2*eps)
        err = abs(d_fd - dw_h[0,0,0,0]) / (abs(d_fd) + 1e-6)
        print("finite-diff dW[0,0,0,0] rel.err:", float(err))
        assert err < 8e-2, f"dW fd mismatch: rel.err={err}"

    # dX only (sanity path)
    dx_d = cp.zeros((N, Cin, H, W), dtype=cp.float32)
    ops_conv2d.backward(
        int(x_d.data.ptr), [N, Cin, H, W],
        int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        None,               # dW
        None,               # dB
        int(dx_d.data.ptr), # dX
        attrs, 0
    )
    dx_h = cp.asnumpy(dx_d)
    assert dx_h.shape == (N, Cin, H, W)
    print("OK: conv2d forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
