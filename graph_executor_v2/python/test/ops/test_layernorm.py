# test_layernorm.py
import os, sys, argparse
import numpy as np

# === Import path & CUDA DLL 경로 (Windows) ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
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
except Exception:
    HAS_CUPY = False
    cp = None

ops_ln = require("layernorm")  # -> _ops_layernorm
LayerNormAttrs = ops_ln.LayerNormAttrs

# ===============================
# Reference implementations (NumPy, CPU)
# ===============================
def ln_forward_ref(x, gamma=None, beta=None, eps=1e-5):
    """
    x: [M,N], gamma/beta: [N] or None
    returns: y, mean, var, xhat
    """
    x = np.asarray(x, dtype=np.float32)
    M, N = x.shape
    mean = x.mean(axis=1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    std  = np.sqrt(var + eps)
    xhat = (x - mean) / std
    if gamma is not None:
        y = xhat * gamma[None, :]
    else:
        y = xhat
    if beta is not None:
        y = y + beta[None, :]
    return y.astype(np.float32), mean.astype(np.float32), var.astype(np.float32), xhat.astype(np.float32)

def ln_backward_ref(x, dy, gamma=None, beta=None, eps=1e-5):
    """
    Analytic gradients for LayerNorm (row-wise). Shapes:
      x, dy: [M,N]; gamma/beta: [N] or None
    Returns: dx, dgamma (or None), dbeta (or None)
    """
    x   = np.asarray(x, dtype=np.float32)
    dy  = np.asarray(dy, dtype=np.float32)
    M, N = x.shape
    y, mean, var, xhat = ln_forward_ref(x, gamma=None, beta=None, eps=eps)  # xhat만 필요
    std = np.sqrt(var + eps)  # [M,1]

    # dgamma / dbeta
    dgamma = None
    dbeta  = None
    if gamma is not None:
        dgamma = np.sum(dy * xhat, axis=0).astype(np.float32)
    if beta is not None:
        dbeta  = np.sum(dy, axis=0).astype(np.float32)

    # dy_hat = dy * gamma (if gamma) else dy
    if gamma is not None:
        dy_hat = dy * gamma[None, :]
    else:
        dy_hat = dy

    # dX (row-wise)
    # dvar, dmu 공식을 사용한 표준 LayerNorm backward
    xm   = x - mean         # [M,N]
    invs = 1.0 / std        # [M,1]

    # dvar = sum(dy_hat * (x-mean) * (-1/2) * (var+eps)^(-3/2))
    dvar = np.sum(dy_hat * xm * (-0.5) * (invs**3), axis=1, keepdims=True)  # [M,1]
    # dmu = sum(dy_hat * -invs) + dvar * sum(-2*(x-mean))/N
    dmu  = np.sum(dy_hat * (-invs), axis=1, keepdims=True) + dvar * np.sum(-2.0 * xm, axis=1, keepdims=True) / N
    # dx = dy_hat * invs + dvar * 2*(x-mean)/N + dmu/N
    dx   = (dy_hat * invs) + (dvar * 2.0 * xm / N) + (dmu / N)

    return dx.astype(np.float32), dgamma, dbeta

# ===============================
# main test
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--finite-diff", action="store_true",
                    help="작은 텐서에서 수치 미분으로 spot-check")
    ap.add_argument("--with-gamma", action="store_true", help="gamma 사용")
    ap.add_argument("--with-beta",  action="store_true", help="beta 사용")
    ap.add_argument("--eps", type=float, default=1e-5)
    args = ap.parse_args()

    print("LOADED:", ops_ln.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_layernorm expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # 작은 사이즈(정확도 체크)
    M, N = 4, 7

    # Host data
    x_h  = rng.standard_normal(size=(M, N), dtype=np.float32)
    dy_h = rng.standard_normal(size=(M, N), dtype=np.float32)

    gamma_h = rng.standard_normal(size=(N,), dtype=np.float32) if args.with_gamma else None
    beta_h  = rng.standard_normal(size=(N,), dtype=np.float32) if args.with_beta  else None

    # Device buffers (CuPy)
    x_d  = cp.asarray(x_h)
    y_d  = cp.zeros((M, N), dtype=cp.float32)
    dy_d = cp.asarray(dy_h)
    dx_d = cp.zeros((M, N), dtype=cp.float32)

    gamma_d = cp.asarray(gamma_h) if gamma_h is not None else None
    beta_d  = cp.asarray(beta_h)  if beta_h  is not None else None

    dgamma_d = cp.zeros((N,), dtype=cp.float32) if gamma_h is not None else None
    dbeta_d  = cp.zeros((N,), dtype=cp.float32) if beta_h  is not None else None

    # Attrs
    attrs = LayerNormAttrs()
    attrs.eps = float(args.eps)

    # ---------- Forward ----------
    ops_ln.forward(
        int(x_d.data.ptr), [M, N],
        (int(gamma_d.data.ptr) if gamma_d is not None else None), [N] if gamma_d is not None else [],
        (int(beta_d.data.ptr)  if beta_d  is not None else None), [N] if beta_d  is not None else [],
        int(y_d.data.ptr), [M, N],
        attrs,
        0
    )
    y_h = cp.asnumpy(y_d)

    # Reference forward
    y_ref, mean_ref, var_ref, xhat_ref = ln_forward_ref(x_h, gamma_h, beta_h, eps=attrs.eps)
    max_abs = float(np.max(np.abs(y_h - y_ref)))
    print("Forward max_abs diff vs ref:", max_abs)
    assert max_abs < 5e-5, f"forward mismatch: {max_abs}"

    # ---------- Backward ----------
    ops_ln.backward(
        int(x_d.data.ptr), [M, N],
        (int(gamma_d.data.ptr) if gamma_d is not None else None), [N] if gamma_d is not None else [],
        int(dy_d.data.ptr), [M, N],
        int(dx_d.data.ptr), [M, N],
        (int(dgamma_d.data.ptr) if dgamma_d is not None else None), [N] if dgamma_d is not None else [],
        (int(dbeta_d.data.ptr)  if dbeta_d  is not None else None), [N] if dbeta_d  is not None else [],
        attrs,
        0
    )
    dx_h      = cp.asnumpy(dx_d)
    dgamma_h  = (cp.asnumpy(dgamma_d) if dgamma_d is not None else None)
    dbeta_h   = (cp.asnumpy(dbeta_d)  if dbeta_d  is not None else None)

    # Reference backward
    dx_ref, dgamma_ref, dbeta_ref = ln_backward_ref(x_h, dy_h, gamma_h, beta_h, eps=attrs.eps)

    max_abs_dx = float(np.max(np.abs(dx_h - dx_ref)))
    print("Backward dX max_abs diff vs ref:", max_abs_dx)
    assert max_abs_dx < 8e-4, f"backward dX mismatch: {max_abs_dx}"

    if gamma_h is not None:
        max_abs_dg = float(np.max(np.abs(dgamma_h - dgamma_ref)))
        print("Backward dgamma max_abs diff:", max_abs_dg)
        assert max_abs_dg < 8e-4, f"dgamma mismatch: {max_abs_dg}"
    if beta_h is not None:
        max_abs_db = float(np.max(np.abs(dbeta_h - dbeta_ref)))
        print("Backward dbeta max_abs diff:", max_abs_db)
        assert max_abs_db < 8e-4, f"dbeta mismatch: {max_abs_db}"

    # ---------- (옵션) finite-difference sanity ----------
    if args.finite_diff:
        # loss = <Y, dy>
        def forward_only(x):
            y, *_ = ln_forward_ref(x, gamma_h, beta_h, eps=attrs.eps)
            return float((y * dy_h).sum())

        eps_fd = 1e-3
        x_pos = x_h.copy(); x_pos[0, 0] += eps_fd
        x_neg = x_h.copy(); x_neg[0, 0] -= eps_fd
        loss_pos = forward_only(x_pos)
        loss_neg = forward_only(x_neg)
        d_fd = (loss_pos - loss_neg) / (2 * eps_fd)
        rel_err = abs(d_fd - dx_ref[0, 0]) / (abs(d_fd) + 1e-6)
        print("finite-diff dX[0,0] rel.err:", float(rel_err))
        assert rel_err < 5e-2, f"finite-diff mismatch: rel.err={rel_err}"

    print("OK: layernorm forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
