# python/test/graph/test_graph_exec_rmsnorm_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import rmsnorm as rn

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
        fn(); cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED"); traceback.print_exc(); return None
    try:
        with s:
            s.begin_capture(); fn(); g = s.end_capture()
        print(f"[{desc}] capture: OK ({type(g).__name__})")
        with s:
            try:
                if hasattr(g, "upload"): g.upload()
            except Exception:
                pass
            g.launch()
        s.synchronize()
        print(f"[{desc}] replay: OK")
        return g
    except Exception:
        print(f"[{desc}] capture/replay: FAILED"); traceback.print_exc(); return None

def test_rmsnorm_fwd_bwd(M=8, N=1024, eps=1e-6, with_affine=True, with_param_grads=True):
    print("\n=== RMSNorm FWD/BWD Capture ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    gamma = cp.random.randn(N).astype(cp.float32) if with_affine else None
    beta  = cp.random.randn(N).astype(cp.float32) if with_affine else None

    Y = cp.empty_like(X)

    def fwd_body():
        out = rn.forward(X, gamma=gamma, beta=beta, eps=eps, out=Y)
        assert out.data.ptr == Y.data.ptr

    dY = cp.random.randn(M, N).astype(cp.float32)
    dX = cp.empty_like(X)
    dgamma = cp.empty((N,), dtype=cp.float32) if (with_affine and with_param_grads) else None
    dbeta  = cp.empty((N,), dtype=cp.float32) if (with_affine and with_param_grads) else None

    def bwd_body():
        dx, dg, db = rn.backward(X, dY, gamma=gamma,
                                 need_dgamma=(dgamma is not None),
                                 need_dbeta=(dbeta is not None),
                                 eps=eps, out=dX)
        assert dx.data.ptr == dX.data.ptr
        if dgamma is not None: dgamma[...] = dg
        if dbeta  is not None: dbeta[...]  = db

    try_capture(fwd_body, "RMSNorm-FWD")
    try_capture(bwd_body, "RMSNorm-BWD")

if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())
    test_rmsnorm_fwd_bwd(M=8, N=2048, eps=1e-6, with_affine=True, with_param_grads=True)
    print("\n[OK] rmsnorm capture tests finished.")
