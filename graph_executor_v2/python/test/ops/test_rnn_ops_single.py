# 0) 제일 먼저: 결정성 환경변수
import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 또는 ":16:8"

# 1) (윈도우) CUDA DLL 경로 추가 등
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp")

# 2) 패키지 경로 세팅
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# 3) 이제 torch 임포트 + 결정성 옵션
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# 4) 여기서 F 임포트하면 됩니다
import torch.nn.functional as F

# 이후 cupy/numpy, 바인딩 모듈 임포트
import cupy as cp
import numpy as np
from graph_executor_v2.ops import _ops_rnn as rnnops



# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_attrs(act="none", with_bias=True, save_z=True, leaky_slope=0.01):
    a = rnnops.RnnAttrs()
    a.with_bias   = bool(with_bias)
    a.save_z      = bool(save_z)
    a.leaky_slope = float(leaky_slope)
    act = act.lower()
    if act == "none":
        a.act = getattr(rnnops.ActKind, "None")
    elif act == "relu":
        a.act = rnnops.ActKind.ReLU
    elif act == "leakyrelu":
        a.act = rnnops.ActKind.LeakyReLU
    elif act == "sigmoid":
        a.act = rnnops.ActKind.Sigmoid
    elif act == "tanh":
        a.act = rnnops.ActKind.Tanh
    elif act == "gelu":
        a.act = rnnops.ActKind.GELU
    else:
        raise ValueError(f"Unknown activation {act}")
    return a


def act_f(x, kind: str, slope=0.01):
    if kind == "none":
        return x
    elif kind == "relu":
        return F.relu(x)
    elif kind == "leakyrelu":
        return F.leaky_relu(x, negative_slope=slope)
    elif kind == "sigmoid":
        return torch.sigmoid(x)
    elif kind == "tanh":
        return torch.tanh(x)
    elif kind == "gelu":
        return F.gelu(x, approximate="tanh")
    else:
        raise ValueError(kind)


# ---------------------------------------------------------------------
# Torch reference (Elman RNN: h_t = act(x_t Wx + h_{t-1} Wh + b))
# ---------------------------------------------------------------------
@torch.no_grad()
def elman_shapes(N, T, I, H):
    return dict(X=(N, T, I), Wx=(I, H), Wh=(H, H), h0=(N, H), Y=(N, T, H), Z=(N, T, H))

def elman_forward_torch(X, Wx, Wh, h0, B, act="relu", slope=0.01, save_z=True):
    """
    X: [N,T,I], Wx:[I,H], Wh:[H,H], h0:[N,H], B:[H] or None
    returns: Y, (optional) Z
    """
    N, T, I = X.shape
    H = Wx.shape[1]
    h_prev = h0
    Y = []
    Z_s = []
    for t in range(T):
        xt = X[:, t, :]                  # [N,I]
        zt = xt @ Wx + h_prev @ Wh       # [N,H]
        if B is not None:
            zt = zt + B.view(1, H)
        yt = act_f(zt, act, slope)
        Y.append(yt)
        if save_z:
            Z_s.append(zt)
        h_prev = yt
    Y = torch.stack(Y, dim=1)            # [N,T,H]
    Z = torch.stack(Z_s, dim=1) if save_z else None
    return Y, Z


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------
def run_case(N=2, T=5, I=8, H=16,
             act="relu", with_bias=True, save_z=True,
             atol=2e-4, rtol=2e-3):

    torch.manual_seed(0)
    cp.random.seed(0)

    print(f"[TEST] RNN N={N}, T={T}, I={I}, H={H}, act={act}, bias={with_bias}, save_z={save_z}")

    # PyTorch reference tensors (requires_grad for backward)
    X_t  = torch.randn((N, T, I), device="cuda", dtype=torch.float32, requires_grad=True)
    Wx_t = torch.randn((I, H),    device="cuda", dtype=torch.float32, requires_grad=True)
    Wh_t = torch.randn((H, H),    device="cuda", dtype=torch.float32, requires_grad=True)
    h0_t = torch.randn((N, H),    device="cuda", dtype=torch.float32, requires_grad=True)
    B_t  = torch.randn((H,),      device="cuda", dtype=torch.float32, requires_grad=True) if with_bias else None

    # Torch forward (and Z for save_z)
    Y_ref, Z_ref = elman_forward_torch(X_t, Wx_t, Wh_t, h0_t, B_t, act, slope=0.01, save_z=True)
    # Backward reference grads
    dY_t = torch.randn_like(Y_ref)
    Y_ref.backward(dY_t)

    dX_ref  = X_t.grad.detach().cpu().numpy()
    dWx_ref = Wx_t.grad.detach().cpu().numpy()
    dWh_ref = Wh_t.grad.detach().cpu().numpy()
    dh0_ref = h0_t.grad.detach().cpu().numpy()
    dB_ref  = B_t.grad.detach().cpu().numpy() if B_t is not None else None

    # ---------------- Prepare CuPy buffers ----------------
    X_cp  = cp.asarray(X_t.detach().contiguous())
    Wx_cp = cp.asarray(Wx_t.detach().contiguous())
    Wh_cp = cp.asarray(Wh_t.detach().contiguous())
    h0_cp = cp.asarray(h0_t.detach().contiguous())
    Y_cp  = cp.zeros((N, T, H), dtype=cp.float32)
    Z_cp  = cp.zeros_like(Y_cp) if save_z else None
    B_cp  = cp.asarray(B_t.detach()) if B_t is not None else None

    # Forward workspaces (required by binding)
    XH_cat = cp.empty((N, I+H), dtype=cp.float32)   # [N,I+H]
    Y_rows = cp.empty((N, H),    dtype=cp.float32)  # [N,H]
    W_cat  = cp.empty((I+H, H),  dtype=cp.float32)  # [I+H,H]
    Z_rows = cp.empty((N, H),    dtype=cp.float32)  # [N,H] (only if save_z)

    attrs = make_attrs(act=act, with_bias=with_bias, save_z=save_z, leaky_slope=0.01)

    # ---------------- Forward ----------------
    rnnops.forward(
        int(X_cp.data.ptr),  [N, T, I],
        int(Wx_cp.data.ptr), [I, H],
        int(Wh_cp.data.ptr), [H, H],
        int(h0_cp.data.ptr), [N, H],
        int(Y_cp.data.ptr),  [N, T, H],
        int(B_cp.data.ptr) if B_cp is not None else None,
        int(Z_cp.data.ptr) if Z_cp is not None else None,
        attrs,
        0,  # stream ptr
        XH_cat_ptr=int(XH_cat.data.ptr),
        Y_rows_ptr=int(Y_rows.data.ptr),
        W_cat_ptr=int(W_cat.data.ptr),
        Z_rows_ptr=int(Z_rows.data.ptr) if save_z else 0
    )

    # Compare forward
    Y_np = cp.asnumpy(Y_cp)
    Y_ref_np = Y_ref.detach().cpu().numpy()
    np.testing.assert_allclose(Y_np, Y_ref_np, atol=atol, rtol=rtol)
    print("✅ forward Y match")

    if save_z:
        Z_np = cp.asnumpy(Z_cp)
        Z_ref_np = Z_ref.detach().cpu().numpy()
        np.testing.assert_allclose(Z_np, Z_ref_np, atol=atol, rtol=rtol)
        print("✅ forward saved Z match")

    # ---------------- Backward ----------------
    dY_cp  = cp.asarray(dY_t.detach().contiguous())
    dWx_cp = cp.zeros_like(Wx_cp)
    dWh_cp = cp.zeros_like(Wh_cp)
    dB_cp  = cp.zeros((H,), dtype=cp.float32) if with_bias else None
    dh0_cp = cp.zeros_like(h0_cp)
    dX_cp  = cp.zeros_like(X_cp)

    # Backward workspaces (all required)
    XH_cat_b = cp.empty((N, I+H), dtype=cp.float32)   # [N,I+H]
    G_rows   = cp.empty((N, H),    dtype=cp.float32)  # [N,H]
    Z_rows_b = cp.empty((N, H),    dtype=cp.float32)  # [N,H]
    W_cat_b  = cp.empty((I+H, H),  dtype=cp.float32)  # [I+H,H]
    dXH_cat  = cp.empty((N, I+H),  dtype=cp.float32)  # [N,I+H]
    dWcat    = cp.empty((I+H, H),  dtype=cp.float32)  # [I+H,H]
    TmpW     = cp.empty((I+H, H),  dtype=cp.float32)  # [I+H,H]

    # Z to consume in bwd: saved or recomputed reference (we already have Z_ref)
    Z_in = Z_cp if save_z else cp.asarray(Z_ref.detach().contiguous())

    rnnops.backward(
        int(X_cp.data.ptr),  [N, T, I],
        int(Wx_cp.data.ptr), [I, H],
        int(Wh_cp.data.ptr), [H, H],
        int(h0_cp.data.ptr), [N, H],
        int(dY_cp.data.ptr), [N, T, H],
        int(Z_in.data.ptr),  [N, T, H],
        int(dWx_cp.data.ptr),
        int(dWh_cp.data.ptr),
        int(dB_cp.data.ptr) if dB_cp is not None else None,
        int(dh0_cp.data.ptr),
        int(dX_cp.data.ptr),
        attrs,
        0,  # stream
        XH_cat_ptr=int(XH_cat_b.data.ptr),
        G_rows_ptr=int(G_rows.data.ptr),
        Z_rows_ptr=int(Z_rows_b.data.ptr),
        W_cat_ptr=int(W_cat_b.data.ptr),
        dXH_cat_ptr=int(dXH_cat.data.ptr),
        dWcat_ptr=int(dWcat.data.ptr),
        TmpW_ptr=int(TmpW.data.ptr)
    )

    # Compare backward
    dX_np  = cp.asnumpy(dX_cp)
    dWx_np = cp.asnumpy(dWx_cp)
    dWh_np = cp.asnumpy(dWh_cp)
    dh0_np = cp.asnumpy(dh0_cp)
    np.testing.assert_allclose(dX_np,  dX_ref,  atol=atol, rtol=rtol)
    np.testing.assert_allclose(dWx_np, dWx_ref, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dWh_np, dWh_ref, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dh0_np, dh0_ref, atol=atol, rtol=rtol)
    print("✅ backward dX/dWx/dWh/dh0 match")

    if with_bias:
        dB_np = cp.asnumpy(dB_cp)
        np.testing.assert_allclose(dB_np, dB_ref, atol=atol, rtol=rtol)
        print("✅ backward dB match")

    print("🎯 RNN forward/backward OK!\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not available.")
    else:
        # 몇 가지 케이스
        run_case(act="none", with_bias=True,  save_z=False)
        run_case(act="relu", with_bias=True,  save_z=True)
        run_case(N=1, T=3, I=4, H=8, act="tanh", with_bias=False, save_z=True)
