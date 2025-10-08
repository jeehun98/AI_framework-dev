# python/test/graph/test_graph_exec_dropout_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import dropout as dop

def env_info():
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name")
        name = name.decode() if isinstance(name, (bytes, bytearray)) else name
        drv = cp.cuda.runtime.driverGetVersion()
        rt  = cp.cuda.runtime.runtimeGetVersion()
        return f"Device: {name}, CC {props.get('major')}.{props.get('minor')} | Driver {drv}, Runtime {rt} | CuPy {cp.__version__}"
    except Exception as e:
        return f"(env info failed: {e})"

def try_capture(fn, desc):
    s = cp.cuda.Stream(non_blocking=True)
    try:
        fn()
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return None
    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
        print(f"[{desc}] capture: OK ({type(g).__name__})")
        with s:
            try:
                if hasattr(g, "upload"):
                    g.upload()
            except Exception:
                pass
            g.launch()
        s.synchronize()
        print(f"[{desc}] replay: OK")
        return g
    except Exception:
        print(f"[{desc}] capture/replay: FAILED")
        traceback.print_exc()
        return None

def test_dropout_forward_and_backward(M=8, N=1024, p=0.2, seed=0xBEEF, counter_base=0):
    print("\n=== Dropout FWD/BWD Capture (stateless RNG) ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    Y = cp.empty_like(X)
    Mask = cp.empty_like(X, dtype=cp.int32)

    # Forward body
    def fwd_body():
        out, m = dop.forward(
            X, p=p, seed=seed, counter_base=counter_base, scale_in_train=True,
            out=Y, out_mask=Mask
        )
        assert out.data.ptr == Y.data.ptr
        assert m is Mask

    # Backward body (uses produced mask)
    dY = cp.random.randn(M, N).astype(cp.float32)
    dX = cp.empty_like(X)
    def bwd_body():
        dx = dop.backward(dY, Mask, p=p, scale_in_train=True, out=dX)
        assert dx.data.ptr == dX.data.ptr

    try_capture(fwd_body, "Dropout-FWD")
    try_capture(bwd_body, "Dropout-BWD")

    # Determinism check under same (seed, counter_base)
    Y2 = cp.empty_like(X); M2 = cp.empty_like(X, dtype=cp.int32)
    def fwd_same():
        dop.forward(X, p=p, seed=seed, counter_base=counter_base, out=Y2, out_mask=M2)
    fwd_same()
    assert cp.allclose(Y, Y2) and cp.all(Mask == M2), "Determinism check failed"

    # Different counter_base should produce different mask
    Y3 = cp.empty_like(X); M3 = cp.empty_like(X, dtype=cp.int32)
    def fwd_diff():
        dop.forward(X, p=p, seed=seed, counter_base=counter_base+12345, out=Y3, out_mask=M3)
    fwd_diff()
    diff_ratio = float(cp.mean((M3 != Mask).astype(cp.float32)))
    print(f"[Dropout] mask diff ratio (different counter_base): {diff_ratio:.3f}")

if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())
    test_dropout_forward_and_backward(M=8, N=2048, p=0.25, seed=0x12345678ABCDEF, counter_base=0)
    print("\n[OK] dropout capture tests finished.")
