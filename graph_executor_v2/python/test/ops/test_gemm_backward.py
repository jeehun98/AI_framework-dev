# python/test/ops/test_gemm_backward.py
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

# graph_executor_v2.ops.require는 네 프로젝트의 동적 로더(= _ops_* 로드 헬퍼)로 가정
from graph_executor_v2.ops import require
ops_gemm = require("gemm")  # -> graph_executor_v2/ops/_ops_gemm.pyd

# _ops_common은 _ops_gemm 내부에서 import하지만, 명시적으로도 한 번 불러 sanity check
try:
    from graph_executor_v2.ops import _ops_common
    HAS_COMMON = True
except Exception:
    HAS_COMMON = False

def list_all_pyd():
    roots = [
        os.path.join(ROOT, "python", "graph_executor_v2", "ops"),
        os.path.dirname(os.__file__),  # ...\Lib
    ]
    found = []
    for base in roots:
        if not os.path.isdir(base):
            continue
        for r, _, files in os.walk(base):
            for f in files:
                if f.startswith("_ops_gemm") and f.endswith(".pyd"):
                    found.append(os.path.join(r, f))
    # sys.path 경로들도 스캔
    for sp in sys.path:
        try:
            if not os.path.isdir(sp):
                continue
            for r, _, files in os.walk(sp):
                for f in files:
                    if f.startswith("_ops_gemm") and f.endswith(".pyd"):
                        p = os.path.join(r, f)
                        if p not in found:
                            found.append(p)
        except Exception:
            pass
    return sorted(set(found))

def check_no_debug_string(pyd_path: str, needle=b"[BWD dbg]"):
    """빌드 바이너리에 디버그 문자열이 남아있는지 단순 확인."""
    try:
        with open(pyd_path, "rb") as f:
            blob = f.read()
        return (needle not in blob)
    except Exception:
        return False  # 읽기 실패 시 실패로 간주

def forward_with_act(A_, B_, bias_, act: str, leaky_slope: float = 0.01):
    """NumPy로 forward 구현 (수치미분 참고용)."""
    M, K = A_.shape
    K2, N = B_.shape
    assert K == K2
    Z_ = A_ @ B_ + bias_.reshape(1, N)
    k = act.lower()
    if k == "relu":
        return np.maximum(Z_, 0)
    if k in ("leakyrelu", "leaky_relu", "lrelu"):
        return np.where(Z_ > 0, Z_, leaky_slope * Z_)
    if k == "tanh":
        return np.tanh(Z_)
    if k == "sigmoid":
        return 1 / (1 + np.exp(-Z_))
    if k == "gelu":
        # tanh-approx
        c = np.sqrt(2/np.pi)
        return 0.5 * Z_ * (1 + np.tanh(c * (Z_ + 0.044715 * (Z_**3))))
    return Z_  # "none"

def finite_diff_check(A, B, bias, act="relu", leaky_slope=0.0, eps=1e-3, tol=8e-2, seed=0):
    """
    간단 수치미분: Z = A@B + bias, Y = act(Z)
    임의 gY ~ N(0,1), analytic gA/gB/gBias vs finite-diff 비교.
    작은 텐서에서만 사용(시간 절약).
    """
    rng = np.random.default_rng(seed)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    Z = A @ B + bias.reshape(1, N)
    gY = rng.standard_normal(size=(M, N)).astype(np.float32)

    out = ops_gemm.backward_numpy(A, B, gY, Z, act=act, bias_kind="pern", leaky_slope=leaky_slope)
    gA, gB, gBias = out["gA"], out["gB"], out["gBias"]
    assert gA.shape == (M, K) and gB.shape == (K, N)

    # gA (i,k) 한 원소 finite-diff
    i, k = 0, 0
    A_pos = A.copy(); A_pos[i, k] += eps
    A_neg = A.copy(); A_neg[i, k] -= eps
    Y_pos = forward_with_act(A_pos, B, bias, act, leaky_slope)
    Y_neg = forward_with_act(A_neg, B, bias, act, leaky_slope)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gA_fd = (loss_pos - loss_neg) / (2 * eps)
    err_gA = abs(gA_fd - gA[i, k]) / (abs(gA_fd) + 1e-6)

    # gB (k,j) 한 원소 finite-diff
    k_, j = 0, 0
    B_pos = B.copy(); B_pos[k_, j] += eps
    B_neg = B.copy(); B_neg[k_, j] -= eps
    Y_pos = forward_with_act(A, B_pos, bias, act, leaky_slope)
    Y_neg = forward_with_act(A, B_neg, bias, act, leaky_slope)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gB_fd = (loss_pos - loss_neg) / (2 * eps)
    err_gB = abs(gB_fd - gB[k_, j]) / (abs(gB_fd) + 1e-6)

    # gBias (perN 가정) 한 원소 finite-diff
    j_ = 0
    bias_pos = bias.copy(); bias_pos[j_] += eps
    bias_neg = bias.copy(); bias_neg[j_] -= eps
    Y_pos = forward_with_act(A, B, bias_pos, act, leaky_slope)
    Y_neg = forward_with_act(A, B, bias_neg, act, leaky_slope)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gBias_fd = (loss_pos - loss_neg) / (2 * eps)
    err_gBias = abs(gBias_fd - (gBias[j_] if gBias is not None else 0.0)) / (abs(gBias_fd) + 1e-6)

    ok = (err_gA < tol) and (err_gB < tol) and (err_gBias < tol)
    return ok, dict(err_gA=float(err_gA), err_gB=float(err_gB), err_gBias=float(err_gBias))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite-diff", action="store_true", help="작은 텐서에서 수치 미분 검증 수행")
    ap.add_argument("--act", type=str, default="relu", choices=["none","relu","leakyrelu","tanh","sigmoid","gelu"])
    ap.add_argument("--leaky-slope", type=float, default=0.0)
    ap.add_argument("--MKN", type=str, default="8,7,5", help="M,K,N (예: 8,7,5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # === 모듈 로드 경로 ===
    print("LOADED:", ops_gemm.__file__)
    if HAS_COMMON:
        print("COMMON_LOADED:", _ops_common.__file__)
    else:
        print("COMMON_LOADED: False (but _ops_gemm should import it internally)")

    # === 중복 pyd 탐지 ===
    all_pyd = list_all_pyd()
    if all_pyd:
        print("FOUND_PYDS:")
        for p in all_pyd:
            mark = " <-- LOADED" if os.path.abspath(p) == os.path.abspath(ops_gemm.__file__) else ""
            print("  ", p, mark)

    # === 로드된 pyd에 디버그 문자열 잔존 확인 ===
    ok_no_dbg = check_no_debug_string(ops_gemm.__file__)
    print("BINARY_HAS_NO_[BWD dbg]:", ok_no_dbg)
    assert ok_no_dbg, "Loaded pyd still contains [BWD dbg] string!"

    # === 랜덤 텐서 준비 ===
    M, K, N = map(int, args.MKN.split(","))
    A = rng.standard_normal(size=(M, K), dtype=np.float32)
    B = rng.standard_normal(size=(K, N), dtype=np.float32)
    bias = rng.standard_normal(size=(N,), dtype=np.float32)

    # === Forward (NumPy helper 경로) ===
    Y = ops_gemm.forward_numpy(A, B, bias, act=args.act, leaky_slope=args.leaky_slope)
    print("Y.shape:", Y.shape)
    assert Y.shape == (M, N)

    # === Backward 준비 (gY, Z) ===
    gY = rng.standard_normal(size=(M, N), dtype=np.float32)
    Z  = (A @ B) + bias.reshape(1, N)

    out = ops_gemm.backward_numpy(A, B, gY, Z,
                                  act=args.act, bias_kind="pern",
                                  leaky_slope=args.leaky_slope)
    print("BACKWARD_KEYS:", list(out.keys()))
    assert out["gC"] is None

    gA, gB, gBias = out["gA"], out["gB"], out["gBias"]
    print("gA.shape:", gA.shape, "gB.shape:", gB.shape)
    if gBias is not None:
        print("gBias.shape:", gBias.shape)
        # pern이면 (N,)
        assert gBias.shape == (N,)

    assert gA.shape == (M, K)
    assert gB.shape == (K, N)

    # === 선택: 수치 미분 체크 ===
    if args.finite_diff:
        print("Running finite-diff check...(small tensors)")
        M2, K2, N2 = 4, 3, 3
        A2 = rng.standard_normal(size=(M2, K2), dtype=np.float32)
        B2 = rng.standard_normal(size=(K2, N2), dtype=np.float32)
        bias2 = rng.standard_normal(size=(N2,), dtype=np.float32)
        ok, errs = finite_diff_check(A2, B2, bias2,
                                     act=args.act,
                                     leaky_slope=args.leaky_slope,
                                     eps=1e-3, tol=8e-2, seed=args.seed+1)
        print("FINITE_DIFF_OK:", ok, errs)
        assert ok, f"Finite-diff mismatch: {errs}"

    print("OK: forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
