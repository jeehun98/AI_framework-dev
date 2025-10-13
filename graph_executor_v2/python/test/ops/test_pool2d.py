# test_pool2d.py
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

# CuPy로 디바이스 포인터 전달/검증
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None

ops_pool = require("pool2d")   # -> _ops_pool2d
Pool2DAttrs = ops_pool.Pool2DAttrs

# ---------------- helpers ----------------
def out_dim_1d(L, k, s, p, d, ceil_mode=False):
    """Conv/Pool 출력 크기 공식 (PyTorch와 동일)"""
    if ceil_mode:
        return int(np.floor((L + 2*p - d*(k-1) - 1 + (s-1)) / s + 1))
    else:
        return int(np.floor((L + 2*p - d*(k-1) - 1) / s + 1))

def out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False):
    Ho = out_dim_1d(H, kH, sH, pH, dH, ceil_mode)
    Wo = out_dim_1d(W, kW, sW, pW, dW, ceil_mode)
    return Ho, Wo

def maxpool2d_ref_forward(x, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False):
    """
    NCHW, float32.
    인덱스는 '입력 평면 절대 인덱스(hi*W + wi)'로 저장.
    패딩은 -inf로 간주(유효 영역만 비교하면 OK).
    """
    N, C, H, W = x.shape
    Ho, Wo = out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)
    y = np.full((N, C, Ho, Wo), -np.inf, dtype=np.float32)
    idx = np.zeros((N, C, Ho, Wo), dtype=np.int32)  # 절대 인덱스 저장

    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                hstart = ho * sH - pH
                for wo in range(Wo):
                    wstart = wo * sW - pW
                    best = -np.inf
                    best_abs = 0
                    for kh in range(kH):
                        hi = hstart + kh * dH
                        if hi < 0 or hi >= H:
                            continue
                        for kw in range(kW):
                            wi = wstart + kw * dW
                            if wi < 0 or wi >= W:
                                continue
                            v = x[n, c, hi, wi]
                            lin_abs = hi * W + wi
                            if v > best:
                                best = v
                                best_abs = lin_abs
                    y[n, c, ho, wo] = best
                    idx[n, c, ho, wo] = best_abs
    return y, idx

def maxpool2d_ref_backward(dy, x, idx, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False):
    """
    argmax 인덱스는 '입력 평면 절대 인덱스(hi*W + wi)'로 해석.
    """
    N, C, H, W = x.shape
    Ho, Wo = dy.shape[2], dy.shape[3]
    dx = np.zeros_like(x, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                for wo in range(Wo):
                    lin_abs = int(idx[n, c, ho, wo])
                    hi = lin_abs // W
                    wi = lin_abs %  W
                    if 0 <= hi < H and 0 <= wi < W:
                        dx[n, c, hi, wi] += dy[n, c, ho, wo]
    return dx

def avgpool2d_ref_forward(x, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False, count_include_pad=False):
    N, C, H, W = x.shape
    Ho, Wo = out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)
    y = np.zeros((N, C, Ho, Wo), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                hstart = ho * sH - pH
                for wo in range(Wo):
                    wstart = wo * sW - pW
                    s = 0.0
                    cnt = 0
                    for kh in range(kH):
                        hi = hstart + kh * dH
                        for kw in range(kW):
                            wi = wstart + kw * dW
                            inside = (0 <= hi < H) and (0 <= wi < W)
                            if inside:
                                s += x[n, c, hi, wi]
                                cnt += 1
                            elif count_include_pad:
                                cnt += 1
                    y[n, c, ho, wo] = s / cnt if cnt else 0.0
    return y

def avgpool2d_ref_backward(dy, x_shape, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode=False, count_include_pad=False):
    N, C, H, W = x_shape
    Ho, Wo = dy.shape[2], dy.shape[3]
    dx = np.zeros(x_shape, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                hstart = ho * sH - pH
                for wo in range(Wo):
                    wstart = wo * sW - pW
                    cnt = 0
                    for kh in range(kH):
                        hi = hstart + kh * dH
                        for kw in range(kW):
                            wi = wstart + kw * dW
                            inside = (0 <= hi < H) and (0 <= wi < W)
                            if inside or count_include_pad:
                                cnt += 1
                    if cnt == 0:
                        continue
                    g = dy[n, c, ho, wo] / cnt
                    for kh in range(kH):
                        hi = hstart + kh * dH
                        for kw in range(kW):
                            wi = wstart + kw * dW
                            if 0 <= hi < H and 0 <= wi < W:
                                dx[n, c, hi, wi] += g
    return dx

def run_case_maxpool(x_h, attrs: Pool2DAttrs, stream=0, check_ref=True):
    assert HAS_CUPY, "CuPy not available"
    N, C, H, W = x_h.shape
    kH, kW = attrs.kH, attrs.kW
    sH, sW = attrs.sH, attrs.sW
    pH, pW = attrs.pH, attrs.pW
    dH, dW = attrs.dH, attrs.dW
    cm     = attrs.ceil_mode

    Ho, Wo = out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, cm)
    x_d = cp.asarray(x_h)
    y_d = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
    ind_d = cp.empty((N, C, Ho, Wo), dtype=cp.int32)

    # forward (with indices)
    ops_pool.maxpool2d_forward(
        int(x_d.data.ptr), [N, C, H, W],
        int(y_d.data.ptr), [N, C, Ho, Wo],
        int(ind_d.data.ptr),
        attrs, stream
    )
    y_h = cp.asnumpy(y_d)
    ind_h = cp.asnumpy(ind_d)

    if check_ref:
        y_ref, idx_ref = maxpool2d_ref_forward(x_h, kH, kW, sH, sW, pH, pW, dH, dW, cm)
        max_abs = float(np.max(np.abs(y_h - y_ref)))
        print(f"  maxpool fwd max_abs: {max_abs:.3e}")
        assert max_abs < 5e-6, f"maxpool forward mismatch: {max_abs}"

    # backward sanity
    rng = np.random.default_rng(0)
    dy_h = rng.standard_normal(size=(N, C, Ho, Wo), dtype=np.float32)
    dy_d = cp.asarray(dy_h)
    dx_d = cp.zeros((N, C, H, W), dtype=cp.float32)

    ops_pool.maxpool2d_backward(
        int(dy_d.data.ptr), [N, C, Ho, Wo],
        int(ind_d.data.ptr), [N, C, Ho, Wo],
        int(dx_d.data.ptr), [N, C, H, W],
        attrs, stream
    )
    dx_h = cp.asnumpy(dx_d)
    if check_ref:
        dx_ref = maxpool2d_ref_backward(dy_h, x_h, ind_h, kH, kW, sH, sW, pH, pW, dH, dW, cm)
        max_abs_dx = float(np.max(np.abs(dx_h - dx_ref)))
        print(f"  maxpool bwd max_abs: {max_abs_dx:.3e}")
        assert max_abs_dx < 5e-6, f"maxpool backward mismatch: {max_abs_dx}"

def run_case_avgpool(x_h, attrs: Pool2DAttrs, stream=0, check_ref=True):
    assert HAS_CUPY, "CuPy not available"
    N, C, H, W = x_h.shape
    kH, kW = attrs.kH, attrs.kW
    sH, sW = attrs.sH, attrs.sW
    pH, pW = attrs.pH, attrs.pW
    dH, dW = attrs.dH, attrs.dW
    cm     = attrs.ceil_mode
    cip    = attrs.count_include_pad

    Ho, Wo = out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, cm)
    x_d = cp.asarray(x_h)
    y_d = cp.empty((N, C, Ho, Wo), dtype=cp.float32)

    # forward
    ops_pool.avgpool2d_forward(
        int(x_d.data.ptr), [N, C, H, W],
        int(y_d.data.ptr), [N, C, Ho, Wo],
        attrs, stream
    )
    y_h = cp.asnumpy(y_d)
    if check_ref:
        y_ref = avgpool2d_ref_forward(x_h, kH, kW, sH, sW, pH, pW, dH, dW, cm, cip)
        max_abs = float(np.max(np.abs(y_h - y_ref)))
        print(f"  avgpool fwd max_abs: {max_abs:.3e}")
        assert max_abs < 5e-6, f"avgpool forward mismatch: {max_abs}"

    # backward
    rng = np.random.default_rng(1)
    dy_h = rng.standard_normal(size=(N, C, Ho, Wo), dtype=np.float32)
    dy_d = cp.asarray(dy_h)
    dx_d = cp.zeros((N, C, H, W), dtype=cp.float32)

    ops_pool.avgpool2d_backward(
        int(dy_d.data.ptr), [N, C, Ho, Wo],
        int(dx_d.data.ptr), [N, C, H, W],
        attrs, stream
    )
    dx_h = cp.asnumpy(dx_d)
    if check_ref:
        dx_ref = avgpool2d_ref_backward(dy_h, (N, C, H, W), kH, kW, sH, sW, pH, pW, dH, dW, cm, cip)
        max_abs_dx = float(np.max(np.abs(dx_h - dx_ref)))
        print(f"  avgpool bwd max_abs: {max_abs_dx:.3e}")
        assert max_abs_dx < 5e-6, f"avgpool backward mismatch: {max_abs_dx}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--mode", choices=["max", "avg", "both"], default="both")
    ap.add_argument("--sweep", action="store_true", help="랜덤 파라미터 스윕")
    args = ap.parse_args()

    print("LOADED:", ops_pool.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_pool2d expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # ===== 기본 소형 케이스 =====
    N, C, H, W = 2, 3, 7, 8
    x_h = rng.standard_normal(size=(N, C, H, W), dtype=np.float32)

    # 공통 attrs 템플릿
    base = Pool2DAttrs()
    base.kH, base.kW = 3, 3
    base.sH, base.sW = 2, 2
    base.pH, base.pW = 1, 1
    base.dH, base.dW = 1, 1
    base.ceil_mode = False
    base.count_include_pad = False

    if args.mode in ("max", "both"):
        print("Case: MaxPool2D basic")
        run_case_maxpool(x_h, base)

    if args.mode in ("avg", "both"):
        print("Case: AvgPool2D basic")
        run_case_avgpool(x_h, base)

    # ===== 랜덤 스윕 (선택) =====
    if args.sweep:
        print("Random sweep...")
        for _ in range(6):
            N, C = int(rng.integers(1, 3)), int(rng.integers(1, 5))
            H, W = int(rng.integers(5, 11)), int(rng.integers(5, 11))
            x_h = rng.standard_normal(size=(N, C, H, W), dtype=np.float32)

            attrs = Pool2DAttrs()
            attrs.kH = int(rng.integers(2, 4)); attrs.kW = int(rng.integers(2, 4))
            attrs.sH = int(rng.integers(1, 3)); attrs.sW = int(rng.integers(1, 3))
            attrs.pH = int(rng.integers(0, 2)); attrs.pW = int(rng.integers(0, 2))
            attrs.dH = 1; attrs.dW = 1
            attrs.ceil_mode = bool(rng.integers(0, 2))
            attrs.count_include_pad = False

            if args.mode in ("max", "both"):
                print(f"  sweep max: NCHW=({N},{C},{H},{W}), k=({attrs.kH},{attrs.kW}), "
                      f"s=({attrs.sH},{attrs.sW}), p=({attrs.pH},{attrs.pW}), ceil={attrs.ceil_mode}")
                run_case_maxpool(x_h, attrs)

            if args.mode in ("avg", "both"):
                print(f"  sweep avg: NCHW=({N},{C},{H},{W}), k=({attrs.kH},{attrs.kW}), "
                      f"s=({attrs.sH},{attrs.sW}), p=({attrs.pH},{attrs.pW}), ceil={attrs.ceil_mode}")
                run_case_avgpool(x_h, attrs)

    print("OK: pool2d forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
