# python/test/bench_gemm.py
import os, sys, time, argparse
import numpy as np
from pathlib import Path

# Make package importable regardless of cwd
THIS = Path(__file__).resolve()
PKG  = str(THIS.parents[1])  # .../python
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# Optional: CUDA DLL hints on Windows
for d in [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
]:
    if os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

from graph_executor_v2 import _core as core

def tflops(M,K,N, sec):
    if sec <= 0: return 0.0
    flops = 2.0 * M * K * N
    return (flops / sec) / 1e12

def run_kernel_only(M,K,N, act="relu", bias="pern", reps=50, warmup=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)
    if bias == "none":
        bias_arr = None
    elif bias == "pern":
        bias_arr = rng.standard_normal((N,), dtype=np.float32)
    elif bias == "perm":
        bias_arr = rng.standard_normal((M,), dtype=np.float32)
    else:
        bias_arr = rng.standard_normal((1,), dtype=np.float32)

    plan = core.GemmPlan(M,K,N, act=act, bias_kind=bias)
    plan.upload(A,B,bias_arr)

    # warmup
    for _ in range(warmup):
        plan.run(copy_out=False)

    times_ms = []
    for _ in range(reps):
        ms = plan.run(copy_out=False)  # kernel-time only
        times_ms.append(ms)

    avg_s = float(np.mean(times_ms))/1000.0
    return dict(M=M,K=K,N=N,act=act,bias=bias,avg_s=avg_s,TFLOPS=tflops(M,K,N,avg_s))

def run_end2end(M,K,N, act="relu", bias="pern", reps=10, warmup=2, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)
    if bias == "none":
        bias_arr = None
    elif bias == "pern":
        bias_arr = rng.standard_normal((N,), dtype=np.float32)
    elif bias == "perm":
        bias_arr = rng.standard_normal((M,), dtype=np.float32)
    else:
        bias_arr = rng.standard_normal((1,), dtype=np.float32)

    # warmup
    for _ in range(warmup):
        _ = core.gemm_bias_act(A,B,bias_arr,act=act)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = core.gemm_bias_act(A,B,bias_arr,act=act)
        t1 = time.perf_counter()
        times.append(t1-t0)

    avg = float(np.mean(times))
    return dict(M=M,K=K,N=N,act=act,bias=bias,avg_s=avg,TFLOPS=tflops(M,K,N,avg))

def main():
    ap = argparse.ArgumentParser(description="GEMM benchmark (kernel-only or end2end)")
    ap.add_argument("--sizes", type=str, default="1024,1024,1024;2048,2048,1024;4096,4096,1024")
    ap.add_argument("--mode", type=str, default="kernel", choices=["kernel","end2end"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bias", type=str, default="pern", choices=["none","pern","perm","scalar"])
    ap.add_argument("--act", type=str, default="relu",
                    choices=["none","relu","leaky_relu","gelu","sigmoid","tanh"])
    args = ap.parse_args()

    results = []
    for triple in args.sizes.split(";"):
        M,K,N = map(int, triple.split(","))
        if args.mode == "kernel":
            r = run_kernel_only(M,K,N, act=args.act, bias=args.bias, reps=args.reps, warmup=args.warmup)
        else:
            r = run_end2end(M,K,N, act=args.act, bias=args.bias, reps=max(10,args.reps//5), warmup=max(2,args.warmup//5))
        results.append(r)
        print(f"{M:5d}x{K:5d}x{N:5d}  mode={args.mode:7s}  act={args.act:10s} bias={args.bias:6s}  "
              f"avg={r['avg_s']:.6f}s  TFLOPS={r['TFLOPS']:.2f}")

    out_dir = Path(__file__).resolve().parents[2] / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bench_gemm_{args.mode}.csv"
    with open(csv_path, "w") as f:
        f.write("M,K,N,mode,act,bias,avg_s,TFLOPS\n")
        for r in results:
            f.write(f"{r['M']},{r['K']},{r['N']},{args.mode},{r['act']},{r['bias']},{r['avg_s']:.6f},{r['TFLOPS']:.3f}\n")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
