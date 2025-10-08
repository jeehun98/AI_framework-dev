import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import rnn as rnn_ops
from graph_executor_v2.ops.rnn import make_ws_fwd_from_arrays, make_ws_bwd_from_arrays
from graph_executor_v2.ops.common import get_stream_ptr

def env_info():
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name  = props.get("name")
        name  = name.decode() if isinstance(name, (bytes, bytearray)) else name
        drv   = cp.cuda.runtime.driverGetVersion()
        rt    = cp.cuda.runtime.runtimeGetVersion()
        return f"Device: {name}, CC {props.get('major')}.{props.get('minor')} | Driver {drv}, Runtime {rt} | CuPy {cp.__version__}"
    except Exception as e:
        return f"(env info failed: {e})"


def try_capture(fn, desc, stream: cp.cuda.Stream):
    """CuPy 13.x: Stream.begin_capture()/end_capture() → Graph.launch()."""
    # warmup on the SAME stream
    try:
        with stream:
            fn(stream)
            cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return None

    # capture + replay
    try:
        with stream:
            stream.begin_capture()
            fn(stream)
            g = stream.end_capture()

        print(f"[{desc}] capture: OK  (graph={type(g).__name__})")

        with stream:
            if hasattr(g, "upload"):
                try:
                    g.upload()
                except Exception:
                    pass
            g.launch()
        stream.synchronize()
        print(f"[{desc}] replay: OK")
        return g
    except Exception:
        print(f"[{desc}] capture/replay: FAILED")
        traceback.print_exc()
        return None


def test_rnn_forward_backward_capture(T=8, B=4, I=16, H=32, use_bias=True, save_z=True):
    print("\n=== RNN Forward/Backward CUDA Graph Capture ===")

    TB = T * B
    # Fixed input/param buffers
    X  = cp.random.randn(TB, I).astype(cp.float32)
    h0 = cp.random.randn(B,  H).astype(cp.float32)
    Wx = cp.random.randn(I,  H).astype(cp.float32)
    Wh = cp.random.randn(H,  H).astype(cp.float32)
    b  = cp.random.randn(H,).astype(cp.float32) if use_bias else None

    # Forward outputs (fixed addresses)
    Hout = cp.empty((TB, H), dtype=cp.float32)
    Zbuf = cp.empty((TB, H), dtype=cp.float32) if save_z else None

    # --- Workspaces (fixed addresses) ---
    # FWD
    PreZ_all = cp.empty((TB, H), dtype=cp.float32)
    TMP_H    = cp.empty((B,  H), dtype=cp.float32)
    TMP_Z    = cp.empty((B,  H), dtype=cp.float32)
    ws_fwd   = make_ws_fwd_from_arrays(PreZ_all, TMP_H, TMP_Z)

    # BWD
    dHsum     = cp.empty((B,  H), dtype=cp.float32)
    dh_next   = cp.empty((B,  H), dtype=cp.float32)
    dZ_all    = cp.empty((TB, H), dtype=cp.float32)
    Hprev_all = cp.empty((TB, H), dtype=cp.float32)
    ws_bwd    = make_ws_bwd_from_arrays(dHsum, dh_next, dZ_all, Hprev_all)

    # Backward outputs (fixed addresses)
    dHout = cp.random.randn(TB, H).astype(cp.float32)
    dX  = cp.empty_like(X)
    dh0 = cp.empty_like(h0)
    dWx = cp.empty_like(Wx)
    dWh = cp.empty_like(Wh)
    dB  = cp.empty((H,), dtype=cp.float32)

    # Dedicated non-default stream for capture
    s = cp.cuda.Stream(non_blocking=True)
    sptr = int(get_stream_ptr(int(s.ptr)))  # rnn_ops.forward/backward expects int or None

    # Forward body (uses SAME buffers & SAME stream)
    def fwd_body(stream):
        rnn_ops.forward(
            X, h0, Wx, Wh,
            b=b, T=T, B=B, save_z=save_z,
            stream=int(get_stream_ptr(int(stream.ptr))),
            out=Hout, zbuf=Zbuf,
            ws_fwd=ws_fwd,
        )

    try_capture(fwd_body, "RNN-Forward", s)

    # Backward body
    def bwd_body(stream):
        rnn_ops.backward(
            X, Hout, h0, Wx, Wh, dHout,
            Zbuf=Zbuf, T=T, B=B,
            stream=int(get_stream_ptr(int(stream.ptr))),
            dX_out=dX, dh0_out=dh0, dWx_out=dWx, dWh_out=dWh, dB_out=dB,
            ws_bwd=ws_bwd,
        )

    try_capture(bwd_body, "RNN-Backward", s)

    # quick sanity checks (no NaNs)
    for name, arr in [("Hout", Hout), ("dX", dX), ("dh0", dh0), ("dWx", dWx), ("dWh", dWh), ("dB", dB)]:
        assert cp.isfinite(arr).all(), f"{name} contains non-finite values"

    print("\n[OK] rnn capture tests finished.")


if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())
    test_rnn_forward_backward_capture(
        T=8, B=4, I=16, H=32, use_bias=True, save_z=True
    )
