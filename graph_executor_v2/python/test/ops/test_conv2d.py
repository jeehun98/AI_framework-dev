# python/test/ops/test_conv2d.py
# =====================================================================================
# Improved conv2d low-level test for _ops_conv2d bindings (NCHW / OIHW, float32)
# - Covers: forward(save_z on/off), backward(dW/dB/dX), workspaces, activations
# - Compares against tiny NumPy reference (slow; only for small shapes)
# - Finite-difference check for a single dW element
# - Requires CuPy (device pointers)
# =====================================================================================

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
ops_conv2d = require("conv2d")  # -> graph_executor_v2/ops/_ops_conv2d.pyd
Conv2DAttrs = ops_conv2d.Conv2DAttrs
ActKind = ops_conv2d.ActKind

# --- CuPy only (we need device pointers) ---
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None


# --------------------------------- utilities ---------------------------------
def ho_wo(H, W, Kh, Kw, sh, sw, ph, pw, dh, dw):
    Ho = (H + 2*ph - dh*(Kh-1) - 1)//sh + 1
    Wo = (W + 2*pw - dw*(Kw-1) - 1)//sw + 1
    return Ho, Wo

def conv2d_ref_nchw(x, w, b=None, stride=(1,1), pad=(0,0), dil=(1,1)):
    """CPU reference conv (NCHW, OIHW), float32, tiny tensors only."""
    N, Cin, H, W = x.shape
    Cout, Cin2, Kh, Kw = w.shape
    assert Cin == Cin2
    sh, sw = stride; ph, pw = pad; dh, dw = dil
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

def make_attrs(sh, sw, ph, pw, dh, dw, with_bias=True, act="none", leaky_slope=0.01, save_z=False):
    attrs = Conv2DAttrs()
    attrs.stride_h, attrs.stride_w = int(sh), int(sw)
    attrs.pad_h, attrs.pad_w       = int(ph), int(pw)
    attrs.dil_h, attrs.dil_w       = int(dh), int(dw)
    attrs.groups                   = 1
    attrs.with_bias                = bool(with_bias)
    attrs.leaky_slope              = float(leaky_slope)
    attrs.save_z                   = bool(save_z)
    attrs.act = getattr(ActKind,
                        "ReLU" if act=="relu" else
                        "LeakyReLU" if act in ("leakyrelu","leaky_relu","lrelu") else
                        "GELU" if act=="gelu" else
                        "Sigmoid" if act=="sigmoid" else
                        "Tanh" if act=="tanh" else
                        "None")
    return attrs

def allocate_workspaces_forward(HWo, K, Cout, need_z_rows: bool):
    dCol   = cp.empty((HWo, K),    dtype=cp.float32)
    W_KC   = cp.empty((K, Cout),   dtype=cp.float32)
    Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
    Z_rows = cp.empty((HWo, Cout), dtype=cp.float32) if need_z_rows else None
    return dCol, W_KC, Y_tmp, Z_rows

def allocate_workspaces_backward(HWo, K, Cout, need_dx: bool, need_dw: bool):
    dCol    = cp.empty((HWo, K),    dtype=cp.float32)  # required

    # (수정) dTmp 2D로, 커널의 ld 가정에 맞춤
    if Cout*K >= HWo*K:
        dTmp = cp.empty((Cout, K), dtype=cp.float32)
    else:
        dTmp = cp.empty((HWo,  K), dtype=cp.float32)

    W_CK    = cp.empty((Cout, K),   dtype=cp.float32) if need_dx else None
    dY_HT   = cp.empty((HWo, Cout), dtype=cp.float32) if need_dx else None
    dWpack  = cp.empty((Cout, K),   dtype=cp.float32) if need_dw else None

    # 문서 기준: (Cout, HWo)
    gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)  # required
    Z_rows  = cp.empty((Cout, HWo), dtype=cp.float32)  # required
    return dCol, dTmp, W_CK, dWpack, dY_HT, gy_rows, Z_rows


# ----------------------------------- runner -----------------------------------
def run_once(args):
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_conv2d expects device pointers.")
        return 0

    rng = np.random.default_rng(args.seed)

    # Shapes (keep tiny if --finite-diff)
    if args.finite_diff:
        N, Cin, H, W   = 2, 3, 6, 5
        Cout, Kh, Kw   = 4, 3, 3
    else:
        # slightly larger but still small
        N, Cin, H, W   = 4, 8, 16, 16
        Cout, Kh, Kw   = 8, 3, 3

    sh, sw = args.stride
    ph, pw = args.pad
    dh, dw = args.dil
    Ho, Wo = ho_wo(H, W, Kh, Kw, sh, sw, ph, pw, dh, dw)
    HWo = Ho * Wo
    K   = Cin * Kh * Kw

    print(f"[shape] X=({N},{Cin},{H},{W}), W=({Cout},{Cin},{Kh},{Kw}), Y/Z=({N},{Cout},{Ho},{Wo})")
    print(f"[im2col] HWo={HWo}, K={K}")

    # Host data
    x_h = rng.standard_normal(size=(N, Cin, H, W), dtype=np.float32)
    w_h = rng.standard_normal(size=(Cout, Cin, Kh, Kw), dtype=np.float32)
    b_h = rng.standard_normal(size=(Cout,), dtype=np.float32)

    # Device buffers
    x_d = cp.asarray(x_h)
    w_d = cp.asarray(w_h)
    b_d = cp.asarray(b_h)
    y_d = cp.zeros((N, Cout, Ho, Wo), dtype=cp.float32)

    # Attrs
    attrs = make_attrs(sh, sw, ph, pw, dh, dw,
                       with_bias=not args.no_bias,
                       act=args.act, leaky_slope=args.leaky_slope,
                       save_z=args.save_z)

    # --- Forward ---
    if attrs.save_z:
        z_d = cp.empty_like(y_d)
        dCol, W_KC, Y_tmp, Z_rows = allocate_workspaces_forward(HWo, K, Cout, need_z_rows=True)
        ops_conv2d.forward(
            int(x_d.data.ptr), [N, Cin, H, W],
            int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
            int(y_d.data.ptr), [N, Cout, Ho, Wo],
            int(b_d.data.ptr) if attrs.with_bias else None,
            int(z_d.data.ptr),           # z_ptr (required since save_z=True)
            attrs,
            0,                           # stream
            int(dCol.data.ptr),
            int(W_KC.data.ptr),
            int(Y_tmp.data.ptr),
            int(Z_rows.data.ptr)         # Z_rows required
        )
        cp.cuda.Stream.null.synchronize()
        print("forward(save_z=True): done")
    else:
        z_d = None
        dCol, W_KC, Y_tmp, Z_rows = allocate_workspaces_forward(HWo, K, Cout, need_z_rows=False)
        ops_conv2d.forward(
            int(x_d.data.ptr), [N, Cin, H, W],
            int(w_d.data.ptr), [Cout, Cin, Kh, Kw],
            int(y_d.data.ptr), [N, Cout, Ho, Wo],
            int(b_d.data.ptr) if attrs.with_bias else None,
            None,                        # z_ptr
            attrs,
            0,
            int(dCol.data.ptr),
            int(W_KC.data.ptr),
            int(Y_tmp.data.ptr),
            0                            # Z_rows_ptr not required
        )
        cp.cuda.Stream.null.synchronize()
        print("forward(save_z=False): done")

    y_h = cp.asnumpy(y_d)
    print("Y.shape:", y_h.shape)
    assert y_h.shape == (N, Cout, Ho, Wo)

    # Quick reference check (pre-activation; act 테스트는 별도 권장)
    if args.ref_check:
        #if attrs.act != ActKind.None:
        #    print("Note: ref_check compares pre-activation only. Set --act none for strict compare.")
        y_ref = conv2d_ref_nchw(x_h, w_h, b_h if attrs.with_bias else None,
                                stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
        max_abs = float(np.max(np.abs(y_h - y_ref)))
        print("Forward max_abs diff vs ref:", max_abs)
        assert max_abs < (5e-4 if args.finite_diff else 3e-3), f"forward mismatch: max_abs={max_abs}"

    # --- Backward quick checks ---
    # Make a random dY; Z is required by backward signature (use saved Z if present or recompute pre-act)
    gy_h = np.random.default_rng(args.seed+1).standard_normal(size=(N, Cout, Ho, Wo)).astype(np.float32)
    gy_d = cp.asarray(gy_h)

    if z_d is None:
        # recompute pre-activation on host for safety (act=none 권장)
        z_h = conv2d_ref_nchw(x_h, w_h, b_h if attrs.with_bias else None,
                              stride=(sh,sw), pad=(ph,pw), dil=(dh,dw)).astype(np.float32)
        z_d = cp.asarray(z_h)

    # dB only
    db_d = cp.zeros((Cout,), dtype=cp.float32)
    # workspaces for backward (no dW/dX here)
    dCol_b, dTmp_b, W_CK_b, dWpack_b, dY_HT_b, gy_rows_b, Z_rows_b = allocate_workspaces_backward(
        HWo, K, Cout, need_dx=False, need_dw=False
    )
    ops_conv2d.backward(
        int(x_d.data.ptr),  [N, Cin, H, W],
        int(w_d.data.ptr),  [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        int(z_d.data.ptr),  [N, Cout, Ho, Wo],
        None,                        # dW
        int(db_d.data.ptr),          # dB
        None,                        # dX
        attrs, 0,
        int(dCol_b.data.ptr),
        int(dTmp_b.data.ptr),
        0,                           # W_CK (not needed)
        0,                           # dWpack (not needed)
        0,                           # dY_HT (not needed)
        int(gy_rows_b.data.ptr),
        int(Z_rows_b.data.ptr)
    )
    cp.cuda.Stream.null.synchronize()
    db_h = cp.asnumpy(db_d)
    db_ref = gy_h.sum(axis=(0, 2, 3))
    max_abs_db = float(np.max(np.abs(db_h - db_ref)))
    print("dB max_abs diff:", max_abs_db)
    assert max_abs_db < (5e-4 if args.finite_diff else 3e-3), f"dB mismatch: {max_abs_db}"

    # dW only
    dw_d = cp.zeros((Cout, Cin, Kh, Kw), dtype=cp.float32)
    dCol_b, dTmp_b, W_CK_b, dWpack_b, dY_HT_b, gy_rows_b, Z_rows_b = allocate_workspaces_backward(
        HWo, K, Cout, need_dx=False, need_dw=True
    )
    ops_conv2d.backward(
        int(x_d.data.ptr),  [N, Cin, H, W],
        int(w_d.data.ptr),  [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        int(z_d.data.ptr),  [N, Cout, Ho, Wo],
        int(dw_d.data.ptr), # dW
        None,               # dB
        None,               # dX
        attrs, 0,
        int(dCol_b.data.ptr),
        int(dTmp_b.data.ptr),
        0,                           # W_CK (not needed)
        int(dWpack_b.data.ptr),      # required for dW path
        0,                           # dY_HT (not needed)
        int(gy_rows_b.data.ptr),
        int(Z_rows_b.data.ptr)
    )
    cp.cuda.Stream.null.synchronize()
    dw_h = cp.asnumpy(dw_d)
    assert dw_h.shape == (Cout, Cin, Kh, Kw)

    # tiny finite-difference for one weight element
    if args.finite_diff:
        # loss = <Y, gy>
        def forward_only(x, w, b):
            y = conv2d_ref_nchw(x, w, b, stride=(sh,sw), pad=(ph,pw), dil=(dh,dw))
            return float((y * gy_h).sum())
        eps = 1e-3
        w_pos = w_h.copy(); w_pos[0,0,0,0] += eps
        w_neg = w_h.copy(); w_neg[0,0,0,0] -= eps
        loss_pos = forward_only(x_h, w_pos, b_h if attrs.with_bias else None)
        loss_neg = forward_only(x_h, w_neg, b_h if attrs.with_bias else None)
        d_fd = (loss_pos - loss_neg) / (2*eps)
        rel_err = abs(d_fd - dw_h[0,0,0,0]) / (abs(d_fd) + 1e-6)
        print("finite-diff dW[0,0,0,0] rel.err:", float(rel_err))
        assert rel_err < 8e-2, f"dW fd mismatch: rel.err={rel_err}"

    # dX only
    dx_d = cp.zeros((N, Cin, H, W), dtype=cp.float32)
    dCol_b, dTmp_b, W_CK_b, dWpack_b, dY_HT_b, gy_rows_b, Z_rows_b = allocate_workspaces_backward(
        HWo, K, Cout, need_dx=True, need_dw=False
    )
    ops_conv2d.backward(
        int(x_d.data.ptr),  [N, Cin, H, W],
        int(w_d.data.ptr),  [Cout, Cin, Kh, Kw],
        int(gy_d.data.ptr), [N, Cout, Ho, Wo],
        int(z_d.data.ptr),  [N, Cout, Ho, Wo],
        None,               # dW
        None,               # dB
        int(dx_d.data.ptr), # dX
        attrs, 0,
        int(dCol_b.data.ptr),
        int(dTmp_b.data.ptr),
        int(W_CK_b.data.ptr),     # required for dX
        0,                        # dWpack not needed
        int(dY_HT_b.data.ptr),    # required for dX
        int(gy_rows_b.data.ptr),
        int(Z_rows_b.data.ptr)
    )
    cp.cuda.Stream.null.synchronize()
    dx_h = cp.asnumpy(dx_d)
    assert dx_h.shape == (N, Cin, H, W)

    print("OK: conv2d forward/backward checks passed.")
    return 0


# ----------------------------------- CLI -----------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--finite-diff", action="store_true",
                    help="Tiny tensors with slow CPU ref + finite difference for a single dW element.")
    ap.add_argument("--ref-check", action="store_true",
                    help="Compare forward output with CPU reference (pre-activation). Use --act none.")
    ap.add_argument("--act", type=str, default="none",
                    choices=["none","relu","leakyrelu","tanh","sigmoid","gelu"])
    ap.add_argument("--leaky-slope", type=float, default=0.01)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--save-z", action="store_true", help="Use Z buffer + Z_rows workspace.")
    ap.add_argument("--stride", type=int, nargs=2, default=[1,1], metavar=("SH","SW"))
    ap.add_argument("--pad",    type=int, nargs=2, default=[1,1], metavar=("PH","PW"))
    ap.add_argument("--dil",    type=int, nargs=2, default=[1,1], metavar=("DH","DW"))
    return ap.parse_args()


def main():
    args = parse_args()
    print("LOADED:", ops_conv2d.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_conv2d expects device pointers.")
        sys.exit(0)
    rc = run_once(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
