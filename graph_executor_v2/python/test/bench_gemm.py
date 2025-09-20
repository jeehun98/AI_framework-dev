import os, sys, time, argparse
import numpy as np

from pathlib import Path

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")

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


def tflops(M,K,N, sec):
    if sec <= 0: return 0.0
    # GEMM FLOPs ~ 2*M*K*N (mul+add), epilogue는 무시(작은 비중)
    flops = 2.0 * M * K * N
    return (flops / sec) / 1e12

def run_one(M,K,N, act="relu", bias="pern", reps=10, warmup=2, seed=0):
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
    ap = argparse.ArgumentParser(description="GEMM forward benchmark")
    ap.add_argument("--sizes", type=str, default="1024,1024,1024;2048,2048,1024",
                    help="semicolon-separated M,K,N triples, e.g. '1024,2048,1024;4096,4096,4096'")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--bias", type=str, default="pern", choices=["none","pern","perm","scalar"])
    ap.add_argument("--act", type=str, default="relu",
                    choices=["none","relu","leaky_relu","gelu","sigmoid","tanh"])
    args = ap.parse_args()

    results = []
    for triple in args.sizes.split(";"):
        M,K,N = map(int, triple.split(","))
        r = run_one(M,K,N, act=args.act, bias=args.bias, reps=args.reps, warmup=args.warmup)
        results.append(r)
        print(f"{M:5d}x{K:5d}x{N:5d}  act={args.act:10s} bias={args.bias:6s}  "
              f"avg={r['avg_s']:.4f}s  TFLOPS={r['TFLOPS']:.2f}")

    # CSV artifact
    out = os.path.join(ROOT, "out")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "bench_gemm.csv")
    with open(csv_path, "w") as f:
        f.write("M,K,N,act,bias,avg_s,TFLOPS\n")
        for r in results:
            f.write(f"{r['M']},{r['K']},{r['N']},{r['act']},{r['bias']},{r['avg_s']:.6f},{r['TFLOPS']:.3f}\n")
    print(f"\nSaved: {csv_path}")

# test_forward_gemm.py 안에 추가
def test_bias_axis_litmus():
    import numpy as np
    from graph_executor_v2 import _core as core
    M,K,N = 256,128,256
    A = np.zeros((M,K), np.float32)
    B = np.zeros((K,N), np.float32)
    bias = np.random.randn(N).astype(np.float32)
    Y = core.gemm_bias_act(A,B,bias,act="none")
    assert np.allclose(Y[0], Y[1])           # PerN이면 행 동일
    assert not np.allclose(Y[:,0], Y[:,1])   # 열은 일반적으로 다름


if __name__ == "__main__":
    # main()
    test_bias_axis_litmus()