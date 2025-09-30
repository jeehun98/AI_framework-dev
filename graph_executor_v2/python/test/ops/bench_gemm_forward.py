# python/test/ops/bench_gemm_forward.py
import os, sys, time, argparse
import numpy as np

# --- repo import path & (Windows) CUDA DLLs ---
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

# --- GPU backend pick: CuPy -> Torch ---
xp = None
cupy = None
torch = None
use_cupy = False
use_torch = False
try:
    import cupy as cp
    cupy = cp
    xp = cp
    use_cupy = True
except Exception:
    try:
        import torch as _torch
        torch = _torch
        xp = _torch
        use_torch = True
    except Exception:
        raise RuntimeError("Need CuPy or PyTorch to run this benchmark.")

from graph_executor_v2.ops import require
ops_gemm = require("gemm")  # loads graph_executor_v2.ops._ops_gemm

# --- helpers ---
def sync():
    if use_cupy:
        cupy.cuda.Stream.null.synchronize()
    else:
        torch.cuda.synchronize()

def ptr(x):
    if use_cupy:
        return int(x.data.ptr)
    return x.data_ptr()

def to_host(x):
    if use_cupy:
        return cupy.asnumpy(x)
    return x.detach().cpu().numpy()

def new(shape, kind="randn"):
    if use_cupy:
        if kind == "zeros": return cupy.zeros(shape, dtype=cupy.float32)
        if kind == "empty": return cupy.empty(shape, dtype=cupy.float32)
        return cupy.random.standard_normal(shape, dtype=cupy.float32)
    else:
        if kind == "zeros": return torch.zeros(*shape, device="cuda", dtype=torch.float32)
        if kind == "empty": return torch.empty(*shape, device="cuda", dtype=torch.float32)
        return torch.randn(*shape, device="cuda", dtype=torch.float32)

def cuda_timer():
    """Return (start, end, elapsed_ms_fn) using CUDA events if available, else wall-clock fallback."""
    if use_cupy:
        start = cupy.cuda.Event()
        end   = cupy.cuda.Event()
        def elapsed_ms():
            end.synchronize()
            return cupy.cuda.get_elapsed_time(start, end)
        return start, end, elapsed_ms
    else:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            def elapsed_ms():
                torch.cuda.synchronize()
                return start.elapsed_time(end)
            return start, end, elapsed_ms
    # fallback (shouldn't happen on GPU path)
    class _Dummy:
        def record(self): pass
    start = _Dummy(); end = _Dummy()
    def elapsed_ms():
        return None
    return start, end, elapsed_ms

def gflops_for_gemm(M,K,N, include_epilogue=True):
    # GEMM: 2*M*K*N FLOPs (mul+add)
    flops = 2.0 * M * K * N
    if include_epilogue:
        # Very rough: bias add + activation per element ~ 2 ops
        flops += 2.0 * M * N
    return flops / 1e9

def run_once_lowlevel(M,K,N, act, leaky):
    # allocate on GPU
    A = new((M,K)); B = new((K,N)); Bias = new((N,))
    Y = new((M,N), kind="empty")

    # wrap as ai::Tensor (row-major contiguous)
    A_t    = ops_gemm.make_tensor_2d(ptr(A),    [M,K])
    B_t    = ops_gemm.make_tensor_2d(ptr(B),    [K,N])
    Bias_t = ops_gemm.make_tensor_2d(ptr(Bias.reshape(1,N) if use_cupy else Bias.view(1,N)), [1,N])
    Y_t    = ops_gemm.make_tensor_2d(ptr(Y),    [M,N])

    attrs = ops_gemm.GemmAttrs()
    attrs.with_bias = True
    attrs.leaky_slope = float(leaky)
    attrs.act = getattr(ops_gemm.ActKind,
                        "ReLU" if act=="relu" else
                        "LeakyReLU" if act in ("leakyrelu","leaky_relu","lrelu") else
                        "GELU" if act=="gelu" else
                        "Sigmoid" if act=="sigmoid" else
                        "Tanh" if act=="tanh" else "None")

    # warmups
    for _ in range(5):
        ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None)
    sync()

    # timed
    start, end, elapsed_ms = cuda_timer()
    if use_cupy:
        start.record()
        ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None)
        end.record()
    else:
        start.record()
        ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None)
        end.record()
    ms = elapsed_ms()
    if ms is None:
        t0 = time.perf_counter()
        ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None)
        sync()
        ms = (time.perf_counter() - t0) * 1e3
    return ms

def run_once_ex(M,K,N, act, leaky):
    # same buffers as lowlevel
    A = new((M,K)); B = new((K,N)); Bias = new((N,))
    Y = new((M,N), kind="empty")
    A_t    = ops_gemm.make_tensor_2d(ptr(A),    [M,K])
    B_t    = ops_gemm.make_tensor_2d(ptr(B),    [K,N])
    Bias_t = ops_gemm.make_tensor_2d(ptr(Bias.reshape(1,N) if use_cupy else Bias.view(1,N)), [1,N])
    Y_t    = ops_gemm.make_tensor_2d(ptr(Y),    [M,N])

    # warmups
    for _ in range(5):
        ops_gemm.forward_ex(A_t, B_t, Bias_t, Y_t,
                            False, False, act, True, float(leaky), None)
    sync()

    # timed
    start, end, elapsed_ms = cuda_timer()
    start.record()
    ops_gemm.forward_ex(A_t, B_t, Bias_t, Y_t,
                        False, False, act, True, float(leaky), None)
    end.record()
    ms = elapsed_ms()
    if ms is None:
        t0 = time.perf_counter()
        ops_gemm.forward_ex(A_t, B_t, Bias_t, Y_t,
                            False, False, act, True, float(leaky), None)
        sync()
        ms = (time.perf_counter() - t0) * 1e3
    return ms

def run_once_numpy(M,K,N, act, leaky):
    # host arrays (NumPy)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)
    bias = rng.standard_normal((N,), dtype=np.float32)

    # warmups
    for _ in range(3):
        _ = ops_gemm.forward_numpy(A,B,bias, act=act, leaky_slope=leaky)

    t0 = time.perf_counter()
    _ = ops_gemm.forward_numpy(A,B,bias, act=act, leaky_slope=leaky)
    ms = (time.perf_counter() - t0) * 1e3
    return ms

def bench(M,K,N, act, leaky, iters):
    # collect multiple runs and report median
    def median_ms(fn):
        xs = []
        for _ in range(iters):
            xs.append(fn(M,K,N,act,leaky))
        xs.sort()
        return xs[len(xs)//2]

    ms_low = median_ms(run_once_lowlevel)
    ms_ex  = median_ms(run_once_ex)
    ms_np  = median_ms(run_once_numpy)

    gflops = gflops_for_gemm(M,K,N, include_epilogue=True)
    return {
        "MKN": (M,K,N),
        "act": act,
        "ms_lowlevel": ms_low,
        "ms_forward_ex": ms_ex,
        "ms_forward_numpy": ms_np,
        "GFLOPs_est": gflops,
        "TeraFLOP/s_low": (gflops / (ms_low/1e3)) / 1e3,
        "TeraFLOP/s_ex":  (gflops / (ms_ex /1e3)) / 1e3,
        "Speedup_low_vs_numpy": ms_np / ms_low if ms_low>0 else float("inf"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="1024,1024,1024;2048,2048,2048",
                    help="semicolon-separated M,K,N triplets, e.g. '1024,1024,1024;4096,4096,4096'")
    ap.add_argument("--act", type=str, default="relu",
                    choices=["none","relu","leakyrelu","tanh","sigmoid","gelu"])
    ap.add_argument("--leaky-slope", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    if use_cupy:
        print("Backend: CuPy")
        cupy.random.seed(123)
    else:
        print("Backend: Torch")
        torch.manual_seed(123)

    sizes = []
    for seg in args.sizes.split(";"):
        M,K,N = map(int, seg.split(","))
        sizes.append((M,K,N))

    print(f"Module: {ops_gemm.__file__}")
    print(f"Act={args.act}, leaky={args.leaky_slope}, iters={args.iters}")
    print("M\tK\tN\tms(low)\tTF/s(low)\tms(ex)\tTF/s(ex)\tms(np)\tspeedup(low/np)")

    for (M,K,N) in sizes:
        res = bench(M,K,N, args.act, args.leaky_slope, args.iters)
        print(f"{M}\t{K}\t{N}\t"
              f"{res['ms_lowlevel']:.3f}\t{res['TeraFLOP/s_low']:.2f}\t"
              f"{res['ms_forward_ex']:.3f}\t{res['TeraFLOP/s_ex']:.2f}\t"
              f"{res['ms_forward_numpy']:.3f}\t{res['Speedup_low_vs_numpy']:.2f}x")

if __name__ == "__main__":
    main()
