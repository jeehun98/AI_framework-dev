# --- path bootstrap (put this at the very top) ---
import os, sys
# CUDA 12.6 경로 예시 — 설치 경로에 맞게 수정
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp")
# cuBLAS 등 추가 DLL이 있다면 (Anaconda/venv에 깔린 cudnn 경로 등) 여기도 추가
# os.add_dll_directory(r"C:\tools\cudnn-9.x-windows\bin")

THIS = os.path.abspath(os.path.dirname(__file__))                 # .../python/test/ops
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))            # .../python
PKG  = os.path.join(ROOT, "python")                               # .../python

# 1) add "<repo>/python" so "graph_executor_v2" becomes importable
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# 2) (Windows) if needed, add CUDA dll dirs here, e.g.:
# import ctypes, glob
# for p in [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"]:
#     if os.path.isdir(p):
#         os.add_dll_directory(p)

# now import through the package path
from graph_executor_v2.ops import _ops_conv2d as convops

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
import torch.nn.functional as F
import cupy as cp


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_attrs(stride=(1,1), pad=(0,0), dil=(1,1),
               with_bias=True, act="none", save_z=False):
    a = convops.Conv2DAttrs()
    a.stride_h, a.stride_w = int(stride[0]), int(stride[1])
    a.pad_h, a.pad_w       = int(pad[0]), int(pad[1])
    a.dil_h, a.dil_w       = int(dil[0]), int(dil[1])
    a.groups               = 1
    a.with_bias            = with_bias
    act = act.lower()
    if act == "none":
        a.act = getattr(convops.ActKind, "None")
    elif act == "relu":
        a.act = convops.ActKind.ReLU
    elif act == "leakyrelu":
        a.act = convops.ActKind.LeakyReLU
    elif act == "sigmoid":
        a.act = convops.ActKind.Sigmoid
    elif act == "tanh":
        a.act = convops.ActKind.Tanh
    elif act == "gelu":
        a.act = convops.ActKind.GELU
    else:
        raise ValueError(f"Unknown activation {act}")
    a.leaky_slope = 0.01
    a.save_z = save_z
    return a


def out_shape(N, Cin, H, W, Cout, Kh, Kw, stride, pad, dil):
    Ho = (H + 2*pad[0] - dil[0]*(Kh-1) - 1)//stride[0] + 1
    Wo = (W + 2*pad[1] - dil[1]*(Kw-1) - 1)//stride[1] + 1
    return N, Cout, Ho, Wo


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------
def run_case(N=2, Cin=8, Cout=16, H=32, W=32, Kh=3, Kw=3,
             stride=(1,1), pad=(1,1), dil=(1,1),
             act="relu", with_bias=True, save_z=True,
             atol=2e-4, rtol=2e-3):

    torch.manual_seed(0)
    cp.random.seed(0)

    print(f"[TEST] Conv2D N={N}, Cin={Cin}, Cout={Cout}, HxW={H}x{W}, "
          f"KhxKw={Kh}x{Kw}, act={act}, bias={with_bias}, save_z={save_z}")

    # PyTorch reference
    x_t = torch.randn((N, Cin, H, W), device="cuda", dtype=torch.float32, requires_grad=True)
    w_t = torch.randn((Cout, Cin, Kh, Kw), device="cuda", dtype=torch.float32, requires_grad=True)
    b_t = torch.randn((Cout,), device="cuda", dtype=torch.float32, requires_grad=True) if with_bias else None

    y_pt = F.conv2d(x_t, w_t, bias=b_t, stride=stride, padding=pad, dilation=dil)
    if act == "relu":
        z_pt = y_pt.clone()
        y_pt = F.relu(y_pt)
    else:
        z_pt = y_pt.clone()

    # Prepare CuPy buffers
    N, Cout, Ho, Wo = out_shape(N,Cin,H,W,Cout,Kh,Kw,stride,pad,dil)
    x_cp = cp.asarray(x_t.detach().contiguous())
    w_cp = cp.asarray(w_t.detach().contiguous())
    y_cp = cp.zeros((N, Cout, Ho, Wo), dtype=cp.float32)
    z_cp = cp.zeros_like(y_cp) if save_z else None
    b_cp = cp.asarray(b_t.detach()) if b_t is not None else None

    # Workspace (forward)
    HWo, K = Ho*Wo, Cin*Kh*Kw
    dCol   = cp.empty((HWo, K), dtype=cp.float32)
    W_KC   = cp.empty((K, Cout), dtype=cp.float32)
    Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
    Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)

    attrs = make_attrs(stride, pad, dil, with_bias, act, save_z)

    # ---------------- Forward ----------------
    convops.forward(
        int(x_cp.data.ptr), [N, Cin, H, W],
        int(w_cp.data.ptr), [Cout, Cin, Kh, Kw],
        int(y_cp.data.ptr), [N, Cout, Ho, Wo],
        int(b_cp.data.ptr) if b_cp is not None else None,
        int(z_cp.data.ptr) if z_cp is not None else None,
        attrs,
        0,
        dCol_ptr=int(dCol.data.ptr),
        W_KC_ptr=int(W_KC.data.ptr),
        Y_tmp_ptr=int(Y_tmp.data.ptr),
        Z_rows_ptr=int(Z_rows.data.ptr) if save_z else 0
    )

    y_np = cp.asnumpy(y_cp)
    y_ref = y_pt.detach().cpu().numpy()
    np.testing.assert_allclose(y_np, y_ref, atol=atol, rtol=rtol)
    print("✅ forward match")

    if save_z:
        z_np = cp.asnumpy(z_cp)
        z_ref = z_pt.detach().cpu().numpy()
        np.testing.assert_allclose(z_np, z_ref, atol=atol, rtol=rtol)
        print("✅ saved Z match")

    # ---------------- Backward ----------------
    dy_t = torch.randn_like(y_pt)
    y_pt.backward(dy_t)

    dx_ref = x_t.grad.detach().cpu().numpy()
    dw_ref = w_t.grad.detach().cpu().numpy()
    db_ref = b_t.grad.detach().cpu().numpy() if b_t is not None else None

    dy_cp = cp.asarray(dy_t.detach())
    dx_cp = cp.zeros_like(x_cp)
    dw_cp = cp.zeros_like(w_cp)
    db_cp = cp.zeros((Cout,), dtype=cp.float32) if with_bias else None
    z_in  = z_cp if save_z else cp.asarray(z_pt.detach())

    # Workspace (backward)
    dCol_b = cp.empty((HWo, K), dtype=cp.float32)
    dTmp   = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
    W_CK   = cp.empty((Cout, K), dtype=cp.float32)
    dWpack = cp.empty((Cout, K), dtype=cp.float32)
    dY_HT  = cp.empty((HWo, Cout), dtype=cp.float32)
    gy_rows= cp.empty((Cout, HWo), dtype=cp.float32)
    Z_rows_b = cp.empty((Cout, HWo), dtype=cp.float32)

    convops.backward(
        int(x_cp.data.ptr), [N, Cin, H, W],
        int(w_cp.data.ptr), [Cout, Cin, Kh, Kw],
        int(dy_cp.data.ptr), [N, Cout, Ho, Wo],
        int(z_in.data.ptr), [N, Cout, Ho, Wo],
        int(dw_cp.data.ptr),
        int(db_cp.data.ptr) if db_cp is not None else None,
        int(dx_cp.data.ptr),
        attrs,
        0,
        dCol_ptr=int(dCol_b.data.ptr),
        dTmp_ptr=int(dTmp.data.ptr),
        W_CK_ptr=int(W_CK.data.ptr),
        dWpack_ptr=int(dWpack.data.ptr),
        dY_HT_ptr=int(dY_HT.data.ptr),
        gy_rows_ptr=int(gy_rows.data.ptr),
        Z_rows_ptr=int(Z_rows_b.data.ptr)
    )

    dx_np = cp.asnumpy(dx_cp)
    dw_np = cp.asnumpy(dw_cp)
    np.testing.assert_allclose(dx_np, dx_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(dw_np, dw_ref, atol=1e-3, rtol=1e-2)
    print("✅ backward dX/dW match")

    if with_bias:
        db_np = cp.asnumpy(db_cp)
        np.testing.assert_allclose(db_np, db_ref, atol=atol, rtol=rtol)
        print("✅ backward dB match")

    print("🎯 Conv2D forward/backward OK!\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not available.")
    else:
        # 간단한 몇 가지 케이스 실행
        run_case(act="none", with_bias=True, save_z=False)
        run_case(act="relu", with_bias=True, save_z=True)
        run_case(N=1, Cin=4, Cout=8, H=15, W=15, Kh=1, Kw=1,
                 act="none", with_bias=False, save_z=False)
