# python/test/ops/test_rnn.py
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

# --- GPU backend pick: CuPy -> Torch (for CUDA events only) ---
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
        raise RuntimeError("Need CuPy or PyTorch for this benchmark.")

# --- framework modules ---
from graph_executor_v2.ops import require
ops_rnn = require("rnn")  # loads python/graph_executor_v2/ops/_ops_rnn

# ---------------- helpers ----------------
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
    class _Dummy:
        def record(self): pass
    start = _Dummy(); end = _Dummy()
    def elapsed_ms(): return None
    return start, end, elapsed_ms

# ---- NumPy reference (vanilla tanh RNN) ----
def numpy_rnn_fwd(XtbI, h0, Wx, Wh, b, T, B, I, H):
    X = XtbI.reshape(T,B,I)
    Hout = np.zeros((T,B,H), dtype=X.dtype)
    Zs   = np.zeros((T,B,H), dtype=X.dtype)
    h = h0.copy()
    for t in range(T):
        z = X[t] @ Wx + h @ Wh + (b if b is not None else 0)
        h = np.tanh(z)
        Hout[t] = h; Zs[t] = z
    return Hout.reshape(T*B, H), Zs.reshape(T*B, H)

def numpy_rnn_bwd(XtbI, HouttbH, ZtbH, h0, Wx, Wh, dHtbH, T,B,I,H):
    X   = XtbI.reshape(T,B,I)
    H_o = HouttbH.reshape(T,B,H)
    Zs  = ZtbH.reshape(T,B,H)
    dH  = dHtbH.reshape(T,B,H)

    dX  = np.zeros_like(X)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db  = np.zeros((H,), dtype=X.dtype)
    dh_next = np.zeros_like(h0)
    for t in reversed(range(T)):
        dht = dH[t] + dh_next
        dzt = dht * (1 - np.tanh(Zs[t])**2)
        db  += dzt.sum(axis=0)
        dWx += X[t].T @ dzt
        htm1 = h0 if t==0 else H_o[t-1]
        dWh += htm1.T @ dzt
        dX[t] = dzt @ Wx.T
        dh_next = dzt @ Wh.T
    return dX.reshape(T*B, I), dh_next, dWx, dWh, db

# ---------------- single forward run (timed) ----------------
def run_forward_once(T,B,I,H, save_z=True, with_bias=True):
    TB = T*B

    # device allocations (row-major 2D)
    X   = new((TB,I))
    h0  = new((B,H))
    Wx  = new((I,H))
    Wh  = new((H,H))
    b   = new((H,)) if with_bias else None
    H_o = new((TB,H), kind="empty")
    Z   = new((TB,H), kind="empty") if save_z else None

    # wrap as ai::Tensor (row-major contiguous)
    X_t   = ops_rnn.make_tensor_2d(ptr(X),   [TB,I])
    h0_t  = ops_rnn.make_tensor_2d(ptr(h0),  [B,H])
    Wx_t  = ops_rnn.make_tensor_2d(ptr(Wx),  [I,H])
    Wh_t  = ops_rnn.make_tensor_2d(ptr(Wh),  [H,H])
    b_t   = ops_rnn.make_tensor_2d(ptr(b.reshape(1,H) if (use_cupy and b is not None) else (b.view(1,H) if b is not None else X)), [1,H]) if b is not None else None
    H_t   = ops_rnn.make_tensor_2d(ptr(H_o), [TB,H])
    Z_t   = ops_rnn.make_tensor_2d(ptr(Z),   [TB,H]) if save_z else None

    attrs = ops_rnn.RNNAttrs()
    attrs.T, attrs.B, attrs.I, attrs.H = T,B,I,H
    attrs.save_z = bool(save_z)

    # warmup
    for _ in range(3):
        ops_rnn.rnn_forward(X_t, h0_t, Wx_t, Wh_t, b_t, H_t, Z_t, attrs)

    sync()
    start, end, elapsed_ms = cuda_timer()
    start.record()
    ops_rnn.rnn_forward(X_t, h0_t, Wx_t, Wh_t, b_t, H_t, Z_t, attrs)
    end.record()
    ms_fwd = elapsed_ms()
    if ms_fwd is None:
        t0 = time.perf_counter()
        ops_rnn.rnn_forward(X_t, h0_t, Wx_t, Wh_t, b_t, H_t, Z_t, attrs)
        sync()
        ms_fwd = (time.perf_counter() - t0) * 1e3

    return ms_fwd, (X,h0,Wx,Wh,b,H_o,Z), (X_t,h0_t,Wx_t,Wh_t,b_t,H_t,Z_t)

# ---------------- end-to-end correctness (vs NumPy) ----------------
def check_correctness(T,B,I,H, atol=1e-4, rtol=1e-4, with_bias=True):
    TB = T*B
    rng = np.random.default_rng(0)
    Xh  = rng.standard_normal((TB,I), dtype=np.float32)
    h0h = rng.standard_normal((B,H),  dtype=np.float32)
    Wxh = rng.standard_normal((I,H),  dtype=np.float32)
    Whh = rng.standard_normal((H,H),  dtype=np.float32)
    bh  = rng.standard_normal((H,),   dtype=np.float32) if with_bias else None
    dHh = rng.standard_normal((TB,H), dtype=np.float32)

    # NumPy ref
    H_np, Z_np = numpy_rnn_fwd(Xh,h0h,Wxh,Whh,bh,T,B,I,H)
    dX_np, dh0_np, dWx_np, dWh_np, db_np = numpy_rnn_bwd(Xh,H_np,Z_np,h0h,Wxh,Whh,dHh,T,B,I,H)

    # device set (copy from host values)
    if use_cupy:
        X   = cupy.asarray(Xh);  h0 = cupy.asarray(h0h)
        Wx  = cupy.asarray(Wxh); Wh = cupy.asarray(Whh)
        b   = cupy.asarray(bh) if bh is not None else None
        dH  = cupy.asarray(dHh)
    else:
        X   = torch.from_numpy(Xh).to("cuda")
        h0  = torch.from_numpy(h0h).to("cuda")
        Wx  = torch.from_numpy(Wxh).to("cuda")
        Wh  = torch.from_numpy(Whh).to("cuda")
        b   = (torch.from_numpy(bh).to("cuda") if bh is not None else None)
        dH  = torch.from_numpy(dHh).to("cuda")

    H_o = new((TB,H), kind="empty")
    Z   = new((TB,H), kind="empty")
    dX  = new((TB,I), kind="empty")
    dh0 = new((B,H),  kind="empty")
    dWx = new((I,H),  kind="empty")
    dWh = new((H,H),  kind="empty")
    dB  = new((H,),   kind="empty")

    # wrap
    X_t   = ops_rnn.make_tensor_2d(ptr(X),   [TB,I])
    h0_t  = ops_rnn.make_tensor_2d(ptr(h0),  [B,H])
    Wx_t  = ops_rnn.make_tensor_2d(ptr(Wx),  [I,H])
    Wh_t  = ops_rnn.make_tensor_2d(ptr(Wh),  [H,H])
    b_t   = ops_rnn.make_tensor_2d(ptr(b.reshape(1,H) if (use_cupy and b is not None) else (b.view(1,H) if b is not None else X)), [1,H]) if b is not None else None
    H_t   = ops_rnn.make_tensor_2d(ptr(H_o), [TB,H])
    Z_t   = ops_rnn.make_tensor_2d(ptr(Z),   [TB,H])
    dH_t  = ops_rnn.make_tensor_2d(ptr(dH),  [TB,H])
    dX_t  = ops_rnn.make_tensor_2d(ptr(dX),  [TB,I])
    dh0_t = ops_rnn.make_tensor_2d(ptr(dh0), [B,H])
    dWx_t = ops_rnn.make_tensor_2d(ptr(dWx), [I,H])
    dWh_t = ops_rnn.make_tensor_2d(ptr(dWh), [H,H])
    dB_t  = ops_rnn.make_tensor_2d(ptr(dB.reshape(1,H) if use_cupy else dB.view(1,H)), [1,H])

    attrs = ops_rnn.RNNAttrs(); attrs.T=T; attrs.B=B; attrs.I=I; attrs.H=H; attrs.save_z=True

    # forward
    ops_rnn.rnn_forward(X_t,h0_t,Wx_t,Wh_t,b_t,H_t,Z_t,attrs)

    # backward
    ops_rnn.rnn_backward(X_t,H_t,Z_t,h0_t,Wx_t,Wh_t,dH_t,
                         dX_t,dh0_t,dWx_t,dWh_t,dB_t,attrs)

    # host compare
    H = to_host(H_o);  Z = to_host(Z)
    dXh  = to_host(dX);  dh0h = to_host(dh0)
    dWxh = to_host(dWx); dWhh = to_host(dWh); dBh = to_host(dB)

    np.testing.assert_allclose(H,   H_np,  rtol=rtol, atol=atol)
    np.testing.assert_allclose(Z,   Z_np,  rtol=rtol, atol=atol)
    np.testing.assert_allclose(dXh, dX_np, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dh0h,dh0_np,rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dWxh,dWx_np,rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dWhh,dWh_np,rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dBh, db_np, rtol=1e-3, atol=1e-3)

# ---------------- tiny bench (forward/backward) ----------------
def bench(T,B,I,H, iters=5, with_bias=True):
    times_f = []
    for _ in range(iters):
        ms, *_ = run_forward_once(T,B,I,H, save_z=True, with_bias=with_bias)
        times_f.append(ms)
    times_f.sort()
    ms_fwd = times_f[len(times_f)//2]

    # backward timing
    _, (X,h0,Wx,Wh,b,H_o,Z), (X_t,h0_t,Wx_t,Wh_t,b_t,H_t,Z_t) = run_forward_once(
        T,B,I,H, save_z=True, with_bias=with_bias
    )
    TB = T*B
    dH = new((TB,H))
    dX  = new((TB,I), kind="empty")
    dh0 = new((B,H),  kind="empty")
    dWx = new((I,H),  kind="empty")
    dWh = new((H,H),  kind="empty")
    dB  = new((H,),   kind="empty")

    dH_t  = ops_rnn.make_tensor_2d(ptr(dH),  [TB,H])
    dX_t  = ops_rnn.make_tensor_2d(ptr(dX),  [TB,I])
    dh0_t = ops_rnn.make_tensor_2d(ptr(dh0), [B,H])
    dWx_t = ops_rnn.make_tensor_2d(ptr(dWx), [I,H])
    dWh_t = ops_rnn.make_tensor_2d(ptr(dWh), [H,H])
    dB_t  = ops_rnn.make_tensor_2d(ptr(dB.reshape(1,H) if use_cupy else dB.view(1,H)), [1,H])

    attrs = ops_rnn.RNNAttrs(); attrs.T=T; attrs.B=B; attrs.I=I; attrs.H=H; attrs.save_z=True

    for _ in range(3):
        ops_rnn.rnn_backward(X_t,H_t,Z_t,h0_t,Wx_t,Wh_t,dH_t,
                             dX_t,dh0_t,dWx_t,dWh_t,dB_t,attrs)
    sync()
    start, end, elapsed_ms = cuda_timer()
    start.record()
    ops_rnn.rnn_backward(X_t,H_t,Z_t,h0_t,Wx_t,Wh_t,dH_t,
                         dX_t,dh0_t,dWx_t,dWh_t,dB_t,attrs)
    end.record()
    ms_bwd = elapsed_ms()
    if ms_bwd is None:
        t0 = time.perf_counter()
        ops_rnn.rnn_backward(X_t,H_t,Z_t,h0_t,Wx_t,Wh_t,dH_t,
                             dX_t,dh0_t,dWx_t,dWh_t,dB_t,attrs)
        sync()
        ms_bwd = (time.perf_counter() - t0) * 1e3

    return ms_fwd, ms_bwd

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="T=64,B=32,I=512,H=512;T=128,B=16,I=1024,H=1024",
                    help="semicolon-separated size sets like 'T=64,B=32,I=512,H=512;T=128,B=16,I=1024,H=1024'")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="run numpy correctness check")
    args = ap.parse_args()

    if use_cupy:
        print("Timer backend: CuPy CUDA events")
        cupy.random.seed(123)
    else:
        print("Timer backend: Torch CUDA events")
        torch.manual_seed(123)

    print(f"Module: {ops_rnn.__file__}")
    print(f"iters={args.iters}")
    print("T\tB\tI\tH\tms(fwd)\tms(bwd)")

    sizes = []
    for seg in args.sizes.split(";"):
        seg = seg.strip()
        kv = dict(tok.split("=") for tok in seg.split(","))
        T,B,I,H = int(kv["T"]), int(kv["B"]), int(kv["I"]), int(kv["H"])
        sizes.append((T,B,I,H))

    for (T,B,I,H) in sizes:
        if args.check:
            check_correctness(T,B,I,H, with_bias=True)
        ms_fwd, ms_bwd = bench(T,B,I,H, iters=args.iters, with_bias=True)
        print(f"{T}\t{B}\t{I}\t{H}\t{ms_fwd:.3f}\t{ms_bwd:.3f}")

if __name__ == "__main__":
    main()
