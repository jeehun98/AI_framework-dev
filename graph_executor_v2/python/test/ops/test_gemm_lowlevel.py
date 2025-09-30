# python/test/ops/test_gemm_lowlevel.py
import os, sys, argparse
import numpy as np

# --- import path / CUDA DLLs (Windows) ---
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

# --- pick GPU array backend: CuPy -> Torch fallback ---
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
        raise RuntimeError("Need CuPy or PyTorch to run low-level GPU test.")

# --- load ops ---
from graph_executor_v2.ops import require
ops_gemm = require("gemm")  # graph_executor_v2/ops/_ops_gemm.pyd

# sanity: common types identity
try:
    from graph_executor_v2.ops import _ops_common as common
    assert ops_gemm.ActKind is common.ActKind
    assert ops_gemm.GemmAttrs is common.GemmAttrs
    assert ops_gemm.Tensor is common.Tensor
    assert ops_gemm.make_tensor_2d is common.make_tensor_2d
    HAS_COMMON = True
except Exception:
    HAS_COMMON = False


def check_no_debug_string(pyd_path: str, needle=b"[BWD dbg]"):
    try:
        with open(pyd_path, "rb") as f:
            return (needle not in f.read())
    except Exception:
        return False


def make_ptr(arr):
    """Return device pointer (int) from CuPy or Torch tensor."""
    if use_cupy:
        return int(arr.data.ptr)
    else:  # torch
        return arr.data_ptr()


def to_numpy(arr):
    if use_cupy:
        return cupy.asnumpy(arr)
    else:
        return arr.detach().cpu().numpy()


def randn(shape, dtype="float32"):
    if use_cupy:
        return cupy.random.standard_normal(shape, dtype=cupy.float32 if dtype=="float32" else None)
    else:
        return torch.randn(*shape, device="cuda", dtype=torch.float32)


def empty(shape, dtype="float32"):
    if use_cupy:
        return cupy.empty(shape, dtype=cupy.float32 if dtype=="float32" else None)
    else:
        return torch.empty(*shape, device="cuda", dtype=torch.float32)


def synchronize():
    if use_cupy:
        cupy.cuda.Stream.null.synchronize()
    else:
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--MKN", type=str, default="64,128,32")
    ap.add_argument("--act", type=str, default="relu", choices=["none","relu","leakyrelu","tanh","sigmoid","gelu"])
    ap.add_argument("--leaky-slope", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--check-numpy", action="store_true", help="numpy helper와 값 비교")
    args = ap.parse_args()

    print("LOADED:", ops_gemm.__file__)
    if HAS_COMMON:
        from graph_executor_v2.ops import _ops_common as common
        print("COMMON:", common.__file__)
    print("BINARY_HAS_NO_[BWD dbg]:", check_no_debug_string(ops_gemm.__file__))

    # shapes
    M, K, N = map(int, args.MKN.split(","))

    # rng
    if use_cupy:
        cupy.random.seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # --- allocate inputs/outputs on GPU ---
    A = randn((M, K))
    B = randn((K, N))
    Bias = randn((N,))        # perN
    Y = empty((M, N))
    gY = randn((M, N))

    # Z = A@B + bias (for backward)
    if use_cupy:
        Z = A @ B + Bias.reshape(1, N)
    else:
        Z = A @ B + Bias.view(1, N)

    # --- wrap as ai::Tensor via make_tensor_2d (zero-copy) ---
    A_t   = ops_gemm.make_tensor_2d(make_ptr(A),   [M, K])
    B_t   = ops_gemm.make_tensor_2d(make_ptr(B),   [K, N])
    Bias_t= ops_gemm.make_tensor_2d(make_ptr(Bias.reshape(1, N) if use_cupy else Bias.view(1, N)), [1, N])
    Y_t   = ops_gemm.make_tensor_2d(make_ptr(Y),   [M, N])

    gY_t  = ops_gemm.make_tensor_2d(make_ptr(gY),  [M, N])
    Z_t   = ops_gemm.make_tensor_2d(make_ptr(Z),   [M, N])

    gA = empty((M, K));  gA_t = ops_gemm.make_tensor_2d(make_ptr(gA), [M, K])
    gB = empty((K, N));  gB_t = ops_gemm.make_tensor_2d(make_ptr(gB), [K, N])
    gBias = empty((N,)); gBias_t = ops_gemm.make_tensor_2d(make_ptr(gBias.reshape(1, N) if use_cupy else gBias.view(1, N)), [1, N])

    # attrs
    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = getattr(ops_gemm.ActKind, "ReLU" if args.act=="relu" else
                        "LeakyReLU" if args.act in ("leakyrelu","leaky_relu","lrelu") else
                        "GELU" if args.act=="gelu" else
                        "Sigmoid" if args.act=="sigmoid" else
                        "Tanh" if args.act=="tanh" else
                        "None")
    attrs.with_bias = True
    attrs.leaky_slope = float(args.leaky_slope)

    # --- forward (low-level) ---
    ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None)
    synchronize()

    print("Y(low-level).shape:", (M, N))

    # --- backward (low-level) ---
    ops_gemm.backward(A_t, B_t, None, gY_t, Z_t, gA_t, gB_t, None, gBias_t, attrs, None)
    synchronize()

    # --- optional: compare with numpy helper path ---
    if args.check_numpy:
        # move to host for comparison
        Y_np     = to_numpy(Y)
        A_np     = to_numpy(A)
        B_np     = to_numpy(B)
        Bias_np  = to_numpy(Bias)
        gY_np    = to_numpy(gY)
        Z_np     = to_numpy(Z)

        Y_ref = ops_gemm.forward_numpy(A_np, B_np, Bias_np, act=args.act, leaky_slope=args.leaky_slope)
        out   = ops_gemm.backward_numpy(A_np, B_np, gY_np, Z_np,
                                        act=args.act, bias_kind="pern",
                                        leaky_slope=args.leaky_slope)
        gA_ref, gB_ref, gBias_ref = out["gA"], out["gB"], out["gBias"]

        atol, rtol = (3e-3, 3e-3) if args.act in ("gelu","tanh","sigmoid") else (1e-3, 1e-3)

        def _allclose(x, y, name):
            ok = np.allclose(x, y, atol=atol, rtol=rtol)
            print(f"compare {name}: {ok} (atol={atol}, rtol={rtol})")
            if not ok:
                diff = np.max(np.abs(x - y))
                print(f"  max abs diff {name}: {diff}")
            assert ok, f"{name} mismatch"

        _allclose(Y_np,    np.asarray(Y_ref), "Y")
        _allclose(to_numpy(gA), gA_ref, "gA")
        _allclose(to_numpy(gB), gB_ref, "gB")
        if gBias_ref is not None:
            _allclose(to_numpy(gBias), gBias_ref, "gBias")

    print("OK: low-level forward/backward works.")

if __name__ == "__main__":
    main()
