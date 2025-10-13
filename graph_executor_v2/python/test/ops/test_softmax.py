# test_softmax.py
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

ops_softmax = require("softmax")  # -> _ops_softmax
SoftmaxAttrs = ops_softmax.SoftmaxAttrs

# 이름 호환 (바인딩이 forward/backward 또는 softmax_forward/backward 인 경우 모두 커버)
FWD = getattr(ops_softmax, "forward", getattr(ops_softmax, "softmax_forward"))
BWD = getattr(ops_softmax, "backward", getattr(ops_softmax, "softmax_backward"))

# ===============================
# Reference implementations (NumPy, CPU)
# ===============================
def _apply_mask(x, mask):
    if mask is None:
        return x
    return x + mask  # numpy broadcasting OK

def softmax_ref(x, mask=None, scale=1.0, axis=-1):
    """Row-wise softmax on 2D [M,N]."""
    z = _apply_mask(scale * x, mask)
    zmax = np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z - zmax)
    y = ez / np.sum(ez, axis=axis, keepdims=True)
    return y

def logsoftmax_ref(x, mask=None, scale=1.0, axis=-1):
    """Row-wise logsoftmax on 2D [M,N]."""
    z = _apply_mask(scale * x, mask)
    zmax = np.max(z, axis=axis, keepdims=True)
    lse  = zmax + np.log(np.sum(np.exp(z - zmax), axis=axis, keepdims=True))
    return z - lse

def softmax_backward_ref(y, dy, scale=1.0, axis=-1):
    """
    dL/dx for y=softmax(scale*x).
    dx = scale * y * (dy - sum(dy * y, axis=axis, keepdims=True))
    """
    dot = np.sum(dy * y, axis=axis, keepdims=True)
    dx = y * (dy - dot)
    return scale * dx

def logsoftmax_backward_ref(y_log, dy, scale=1.0, axis=-1):
    """
    dL/dx for y_log=logsoftmax(scale*x).
    Let p = exp(y_log). Then:
      dx = scale * ( dy - p * sum(dy, axis=axis, keepdims=True) )
    """
    p = np.exp(y_log)
    s = np.sum(dy, axis=axis, keepdims=True)
    dx = dy - p * s
    return scale * dx

def make_mask(shape, kind="row", rng=None):
    """
    kind: 'none' | 'row' | 'col' | 'full'
      - row: [M,1]
      - col: [1,N]
      - full:[M,N]
    mask 값은 일부 위치에 큰 음수(-1e9) 부여해서 마스킹
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M, N = shape
    if kind == "none":
        return None
    if kind == "row":
        m = np.zeros((M, 1), dtype=np.float32)
        # 각 row에서 30% 확률로 한 칸 마스킹 (예시)
        for i in range(M):
            if rng.random() < 0.3:
                m[i, 0] = -1e9
        return m
    if kind == "col":
        m = np.zeros((1, N), dtype=np.float32)
        for j in range(N):
            if rng.random() < 0.3:
                m[0, j] = -1e9
        return m
    if kind == "full":
        m = np.zeros((M, N), dtype=np.float32)
        # 10% 위치 마스킹
        mask_pick = rng.random((M, N)) < 0.1
        m[mask_pick] = -1e9
        return m
    raise ValueError("unknown mask kind")

# ===============================
# main test
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--finite-diff", action="store_true",
                    help="작은 텐서에서 수치 미분/참조 구현과 비교")
    ap.add_argument("--log", action="store_true",
                    help="logsoftmax 경로로 테스트 (기본은 softmax)")
    ap.add_argument("--mask", choices=["none", "row", "col", "full"], default="none")
    ap.add_argument("--scale", type=float, default=0.7, help="temperature inverse (y=softmax(scale*x))")
    args = ap.parse_args()

    print("LOADED:", ops_softmax.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_softmax expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # 작은 사이즈(정확도 체크)
    M, N = 4, 7

    # Host data
    x_h  = rng.standard_normal(size=(M, N), dtype=np.float32)
    dy_h = rng.standard_normal(size=(M, N), dtype=np.float32)

    # Mask
    mask_h = None if args.mask == "none" else make_mask((M, N), args.mask, rng=rng)

    # Device buffers (CuPy)
    x_d  = cp.asarray(x_h)
    y_d  = cp.zeros((M, N), dtype=cp.float32)
    dy_d = cp.asarray(dy_h)
    dx_d = cp.zeros((M, N), dtype=cp.float32)
    mask_d = None
    mask_tuple = None
    if mask_h is not None:
        mask_d = cp.asarray(mask_h)
        mask_tuple = (int(mask_d.data.ptr), list(mask_h.shape))

    # Attrs
    attrs = SoftmaxAttrs()
    attrs.scale = float(args.scale)
    attrs.log   = bool(args.log)

    # ---------- Forward ----------
    FWD(
        int(x_d.data.ptr), [M, N],
        int(y_d.data.ptr), [M, N],
        mask_tuple if mask_tuple is not None else None,
        attrs,
        0  # stream=0 (default)
    )
    y_h = cp.asnumpy(y_d)

    # Reference forward
    if not args.log:
        y_ref = softmax_ref(x_h, mask=mask_h, scale=attrs.scale, axis=1)
        # 확률합 체크
        row_sum = np.sum(y_h, axis=1)
        print("Row sum (softmax):", row_sum)
        assert np.allclose(row_sum, 1.0, atol=5e-5), "Row sums not ~1 for softmax"
    else:
        y_ref = logsoftmax_ref(x_h, mask=mask_h, scale=attrs.scale, axis=1)
        # logsoftmax 검증: exp(y)의 합=1
        row_sum = np.sum(np.exp(y_h), axis=1)
        print("Row sum (exp(logsoftmax)):", row_sum)
        assert np.allclose(row_sum, 1.0, atol=5e-5), "Row sums(exp(logsoftmax)) not ~1"

    max_abs = float(np.max(np.abs(y_h - y_ref)))
    print("Forward max_abs diff vs ref:", max_abs)
    assert max_abs < 5e-4, f"forward mismatch: {max_abs}"

    # ---------- Backward ----------
    # y_or_x: 권장은 y(재계산 방지). logsoftmax도 동일하게 y_or_x=Y(logits 아님)
    y_or_x_d = y_d  # y_provided=True
    BWD(
        int(y_or_x_d.data.ptr), [M, N],
        int(dy_d.data.ptr),     [M, N],
        int(dx_d.data.ptr),     [M, N],
        mask_tuple if mask_tuple is not None else None,
        attrs,
        True,   # y_provided
        0
    )
    dx_h = cp.asnumpy(dx_d)

    # Reference backward
    if not args.log:
        # y_ref는 softmax 출력
        dx_ref = softmax_backward_ref(y_ref, dy_h, scale=attrs.scale, axis=1)
    else:
        # y_ref는 logsoftmax 출력
        dx_ref = logsoftmax_backward_ref(y_ref, dy_h, scale=attrs.scale, axis=1)

    max_abs_dx = float(np.max(np.abs(dx_h - dx_ref)))
    print("Backward dX max_abs diff vs ref:", max_abs_dx)
    assert max_abs_dx < 8e-4, f"backward mismatch: {max_abs_dx}"

    # ---------- (옵션) finite difference sanity on one element ----------
    if args.finite_diff:
        eps = 1e-3
        # loss = <Y_or_logY, dy> (logsoftmax일 때도 동일하게 <logY, dy>)
        def forward_only(x):
            if args.log:
                y = logsoftmax_ref(x, mask=mask_h, scale=attrs.scale, axis=1)
            else:
                y = softmax_ref(x,      mask=mask_h, scale=attrs.scale, axis=1)
            return float((y * dy_h).sum())

        x_pos = x_h.copy(); x_pos[0, 0] += eps
        x_neg = x_h.copy(); x_neg[0, 0] -= eps
        loss_pos = forward_only(x_pos)
        loss_neg = forward_only(x_neg)
        d_fd = (loss_pos - loss_neg) / (2 * eps)

        # analytic grad at (0,0)
        d_an = dx_ref[0, 0]
        rel_err = abs(d_fd - d_an) / (abs(d_fd) + 1e-6)
        print("finite-diff dX[0,0] rel.err:", float(rel_err))
        assert rel_err < 8e-2, f"finite-diff mismatch: rel.err={rel_err}"

    print("OK: softmax/logsoftmax forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
