import os, sys, time, argparse, statistics as stats
import numpy as np

# Make project python package importable if script is run from anywhere
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from pathlib import Path

# 1) graph_executor_v2/python 을 import path 에 추가
PKG = str(Path(__file__).resolve().parents[1])
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# 2) 필요한 DLL 폴더들을 미리 등록 (Python 3.8+)
dll_hints = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",  # cudart64_126.dll, cublas*.dll 등
    # 빌드 산출물 폴더에 별도 .dll 이 있다면 여기도 추가
    # r"C:\Users\as042\Desktop\AI_framework-dev\graph_executor_v2\build\Release",
    # VS CRT이 필요할 경우(대개는 불필요하지만 혹시):
    # r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\x64\Microsoft.VC143.CRT",
]
for d in dll_hints:
    if os.path.isdir(d):
        os.add_dll_directory(d)

from graph_executor_v2 import _core as core



def gelu_tanh(x: np.ndarray) -> np.ndarray:
    k0 = 0.7978845608
    k1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(k0 * (x + k1 * x**3)))

def apply_act(x: np.ndarray, act: str, leaky: float = 0.01) -> np.ndarray:
    act = act.lower()
    if act in ("none", "identity"):
        return x
    if act == "relu":
        return np.maximum(x, 0.0)
    if act in ("leakyrelu","leaky_relu","lrelu"):
        y = x.copy()
        y[y<0] *= leaky
        return y
    if act == "gelu":
        return gelu_tanh(x)
    if act == "sigmoid":
        return 1.0/(1.0+np.exp(-x))
    if act == "tanh":
        return np.tanh(x)
    raise ValueError(f"unknown act {act}")

def make_bias(kind, M, N, rng):
    if kind == "none":
        return None
    if kind == "scalar":
        return rng.standard_normal((1,), dtype=np.float32)
    if kind == "perm":
        return rng.standard_normal((M,), dtype=np.float32)
    if kind == "pern":
        return rng.standard_normal((N,), dtype=np.float32)
    raise ValueError("bias_kind must be one of: none, scalar, perm, pern")

def run_case(M, K, N, act="relu", bias_kind="pern", seed=0, reps=5, warmup=1, tol=5e-5, verbose=False):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)
    bias = make_bias(bias_kind, M, N, rng)

    # Reference
    Z = A @ B

    if bias is not None:
        if bias.shape == (1,):
            Z = Z + bias[0]
        elif bias.shape == (N,):          # ← PerN을 먼저!
            Z = Z + bias.reshape(1, N)
        elif bias.shape == (M,):
            Z = Z + bias.reshape(M, 1)

    Y_ref = apply_act(Z, act)

    # Warmup
    for _ in range(max(0, warmup)):
        _ = core.gemm_bias_act(A, B, bias, act=act)

    # Timed runs
    times = []
    Y = None
    for _ in range(reps):
        t0 = time.perf_counter()
        Y = core.gemm_bias_act(A, B, bias, act=act)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Compare
    max_abs = float(np.max(np.abs(Y - Y_ref)))
    denom = float(np.max(np.abs(Y_ref)) + 1e-6)
    rel = max_abs / denom
    ok = (max_abs <= tol)

    if verbose or not ok:
        print(f"[{M}x{K}]x[{K}x{N}] act={act:10s} bias={bias_kind:6s}  "
              f"max_abs={max_abs:.3e} rel={rel:.3e}  "
              f"time(avg/p50/p95)={np.mean(times):.4f}/{np.median(times):.4f}/{np.percentile(times,95):.4f}s  "
              f"reps={reps}")
    return ok, max_abs, rel, times

def main():
    ap = argparse.ArgumentParser(description="Forward GEMM+bias+act tests for graph_executor_v2")
    ap.add_argument("--tol", type=float, default=5e-5, help="absolute tolerance")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # test matrix
    cases = [
        (16, 32,  8,  "none",       "none"),
        (32, 64, 16,  "relu",       "pern"),
        (64, 64, 64,  "gelu",       "scalar"),
        (31, 17, 29,  "tanh",       "perm"),
        (128,128,64,  "leaky_relu", "pern"),
        (8,   8,  8,  "sigmoid",    "none"),
        (256, 128, 256, "relu",     "pern"),
    ]

    ok_all = True
    report = []
    for (M,K,N,act,bk) in cases:
        ok, ma, rel, times = run_case(M,K,N, act=act, bias_kind=bk,
                                      seed=args.seed, reps=args.reps,
                                      warmup=args.warmup, tol=args.tol,
                                      verbose=True or args.verbose)
        report.append(dict(M=M,K=K,N=N,act=act,bias=bk,max_abs=ma,rel=rel,
                           t_avg=float(np.mean(times)),
                           t_p50=float(np.median(times)),
                           t_p95=float(np.percentile(times,95))))
        ok_all &= ok

    # Pretty summary
    print("\n=== SUMMARY ===")
    for r in report:
        print(f"{r['M']:4d}x{r['K']:4d}x{r['N']:4d} act={r['act']:10s} bias={r['bias']:6s}  "
              f"max_abs={r['max_abs']:.2e}  avg={r['t_avg']:.4f}s")

    # Optional JSON artifact (CI, dashboards, etc.)
    out_dir = os.environ.get("GE2_OUT", os.path.join(ROOT, "out"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "forward_report.json"), "w") as f:
        import json; json.dump(report, f, indent=2)

    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    # Optional: block for easier debugging
    print("PYTHONPATH[0]:", sys.path[0])
    print("_core file    :", core.__file__)

    if os.environ.get("CUDA_LAUNCH_BLOCKING","0") == "1":
        pass
    main()
