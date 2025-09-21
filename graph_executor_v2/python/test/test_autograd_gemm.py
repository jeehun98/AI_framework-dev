# python/test/test_autograd_gemm.py
import os, sys, argparse
import numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# CUDA DLL (Windows) 힌트 경로
cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2 import _core as core


# === NumPy 참조 연산 ===
def gelu_tanh(x: np.ndarray) -> np.ndarray:
    k0 = 0.7978845608
    k1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(k0 * (x + k1 * x**3)))

def apply_act(x: np.ndarray, act: str, leaky: float = 0.01) -> np.ndarray:
    a = act.lower()
    if a in ("none", "identity"):  return x
    if a == "relu":                return np.maximum(x, 0.0)
    if a in ("leakyrelu","leaky_relu","lrelu"):
        y = x.copy()
        y[y<0] *= leaky
        return y
    if a == "gelu":                return gelu_tanh(x)
    if a == "sigmoid":             return 1.0/(1.0+np.exp(-x))
    if a == "tanh":                return np.tanh(x)
    raise ValueError(f"unknown act {act}")

def make_bias(kind: str, M: int, N: int, rng) -> np.ndarray | None:
    k = kind.lower()
    if k == "none":   return None
    if k == "scalar": return rng.standard_normal((1,), dtype=np.float32)
    if k == "perm":   return rng.standard_normal((M,), dtype=np.float32)
    if k == "pern":   return rng.standard_normal((N,), dtype=np.float32)
    raise ValueError("bias_kind must be one of: none, scalar, perm, pern")

def forward_numpy(A, B, bias, act: str, leaky=0.01):
    Z = A @ B
    if bias is not None:
        if bias.shape == (1,):
            Z = Z + bias[0]
        elif bias.shape == (A.shape[0],):
            Z = Z + bias.reshape(-1,1)
        elif bias.shape == (B.shape[1],):
            Z = Z + bias.reshape(1,-1)
    Y = apply_act(Z, act, leaky)
    return Y, Z

def loss_numpy(A, B, bias, gY, act, leaky=0.01):
    Y, _ = forward_numpy(A, B, bias, act, leaky)
    return float(np.sum(gY * Y))


# === 유틸: 수치 미분(중심차분) ===
def numeric_grad_A(A, B, bias, gY, act, eps=1e-3, leaky=0.01):
    M, K = A.shape
    G = np.zeros_like(A, dtype=np.float32)
    for i in range(M):
        for k in range(K):
            old = A[i,k]
            A[i,k] = old + eps
            Lp = loss_numpy(A,B,bias,gY,act,leaky)
            A[i,k] = old - eps
            Lm = loss_numpy(A,B,bias,gY,act,leaky)
            A[i,k] = old
            G[i,k] = (Lp - Lm)/(2*eps)
    return G

def numeric_grad_B(A, B, bias, gY, act, eps=1e-3, leaky=0.01):
    K, N = B.shape
    G = np.zeros_like(B, dtype=np.float32)
    for k in range(K):
        for j in range(N):
            old = B[k,j]
            B[k,j] = old + eps
            Lp = loss_numpy(A,B,bias,gY,act,leaky)
            B[k,j] = old - eps
            Lm = loss_numpy(A,B,bias,gY,act,leaky)
            B[k,j] = old
            G[k,j] = (Lp - Lm)/(2*eps)
    return G

def numeric_grad_bias(A,B,bias, gY, act, eps=1e-3, leaky=0.01):
    if bias is None:
        return None
    g = np.zeros_like(bias, dtype=np.float32)
    for i in range(bias.shape[0]):
        old = bias[i]
        bias[i] = old + eps
        Lp = loss_numpy(A,B,bias,gY,act,leaky)
        bias[i] = old - eps
        Lm = loss_numpy(A,B,bias,gY,act,leaky)
        bias[i] = old
        g[i] = (Lp - Lm)/(2*eps)
    return g

def compare_grads(name, ana, num, rel_tol=1e-2, abs_tol=5e-3, verbose=True):
    diff = ana - num
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(1e-6, np.abs(num))
    rel = np.max(np.abs(diff) / denom) if diff.size else 0.0
    ok = (max_abs <= abs_tol) or (rel <= rel_tol)
    if verbose:
        print(f"  {name:6s}  max_abs={max_abs:.3e}  rel_max={rel:.3e}  -> {'OK' if ok else 'BAD'}")
    return ok, max_abs, rel


def run_case(M, K, N, act="relu", bias_kind="pern",
             seed=123, leaky=0.01, eps=1e-3,
             fwd_tol=5e-5, rel_tol=1e-2, abs_tol=5e-3, verbose=True):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)
    gY = rng.standard_normal((M,N), dtype=np.float32)
    bias = make_bias(bias_kind, M, N, rng)

    # 1) FWD 정확도 (NumPy vs CUDA)
    Y_ref, Z_ref = forward_numpy(A,B,bias, act, leaky)
    Y_cuda = core.gemm_bias_act(A, B, bias, act=act, leaky_slope=leaky)
    max_abs = float(np.max(np.abs(Y_cuda - Y_ref)))
    rel = max_abs / (float(np.max(np.abs(Y_ref))) + 1e-6)
    ok_fwd = (max_abs <= fwd_tol)
    if verbose:
        print(f"[{M}x{K}]x[{K}x{N}] act={act:10s} bias={bias_kind:6s}  "
              f"FWD max_abs={max_abs:.3e} rel={rel:.3e}  -> {'OK' if ok_fwd else 'BAD'}")

    # 2) FWD-with-Z 검증: Y == act(Z)
    Y2, Z = core.gemm_bias_act_fwd_with_z(A, B, bias, act=act, leaky_slope=leaky)
    y_from_z = apply_act(Z, act, leaky)
    ok_z = float(np.max(np.abs(Y2 - y_from_z))) <= fwd_tol
    if verbose:
        print(f"  Z-stash check: ||Y - act(Z)||_inf = {float(np.max(np.abs(Y2 - y_from_z))):.3e}  -> {'OK' if ok_z else 'BAD'}")

    # 3) BWD (CUDA analytic)
    out = core.gemm_bias_act_bwd(A, B, gY, Z, act=act, bias_kind=bias_kind, leaky_slope=leaky)
    gA_ana = out["gA"]
    gB_ana = out["gB"]
    gBias_ana = out["gBias"] if out["gBias"] is not None else None

    # 4) 수치 미분 (NumPy 중심차분)
    A_num = A.copy(); B_num = B.copy()
    bias_num = None if bias is None else bias.copy()
    gA_num = numeric_grad_A(A_num, B_num, bias_num, gY, act, eps=eps, leaky=leaky)
    gB_num = numeric_grad_B(A, B, bias_num, gY, act, eps=eps, leaky=leaky)
    gBias_num = numeric_grad_bias(A, B, bias_num, gY, act, eps=eps, leaky=leaky)

    # 5) 비교
    okA, maA, reA = compare_grads("gA", gA_ana, gA_num, rel_tol, abs_tol, verbose)
    okB, maB, reB = compare_grads("gB", gB_ana, gB_num, rel_tol, abs_tol, verbose)
    if gBias_ana is None and gBias_num is None:
        okBias, maBias, reBias = True, 0.0, 0.0
        if verbose: print("  gBias  (not used)")
    else:
        okBias, maBias, reBias = compare_grads("gBias", gBias_ana, gBias_num, rel_tol, abs_tol, verbose)

    ok_all = ok_fwd and ok_z and okA and okB and okBias
    if verbose:
        print("  =>", "PASS" if ok_all else "FAIL")
    return ok_all


def main():
    ap = argparse.ArgumentParser("graph_executor_v2 GEMM autograd test")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--leaky", type=float, default=0.01)
    ap.add_argument("--fwd_tol", type=float, default=5e-5)
    ap.add_argument("--abs_tol", type=float, default=5e-3)
    ap.add_argument("--rel_tol", type=float, default=1e-2)
    args = ap.parse_args()

    cases = [
        (8,7,5,"none",       "none"),
        (8,7,5,"relu",       "pern"),
        (8,7,5,"leaky_relu", "perm"),
        (8,7,5,"gelu",       "scalar"),
        (8,7,5,"sigmoid",    "none"),
        (8,7,5,"tanh",       "pern"),
    ]

    ok_all = True
    for (M,K,N,act,bk) in cases:
        print(f"\n=== CASE: M={M} K={K} N={N} act={act} bias={bk} ===")
        ok = run_case(M,K,N, act=act, bias_kind=bk,
                      seed=args.seed, leaky=args.leaky, eps=args.eps,
                      fwd_tol=args.fwd_tol, abs_tol=args.abs_tol, rel_tol=args.rel_tol, verbose=True)
        ok_all &= ok

    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    # 바인딩 파일 경로 출력(디버깅용)
    print("PYTHONPATH[0]:", sys.path[0])
    print("_core file    :", core.__file__)
    main()
