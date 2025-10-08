# python/test/graph/test_graph_exec_layernorm_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import layernorm as ln_ops

# ======================== env ========================
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


# ===================== capture utils ====================
def try_capture(fn, desc):
    """desc 설명과 함께 fn()을 캡쳐해 본다. CuPy 13.x: Graph.launch() 사용."""
    s = cp.cuda.Stream(non_blocking=True)

    # 1) warmup
    try:
        fn()
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return None

    # 2) capture
    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
        print(f"[{desc}] capture: OK  (graph={type(g).__name__})")

        # 3) replay (instantiate 없음 → upload/launch 직접)
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


# ======================== tests =========================
def test_layernorm_forward_no_affine(M=8, N=1024, eps=1e-5):
    print("\n=== LayerNorm FWD (no affine) Capture ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    Y = cp.empty_like(X)

    def body():
        ln_ops.forward(X, gamma=None, beta=None, eps=eps, out=Y)

    try_capture(body, "LayerNorm-FWD(no-affine)")


def test_layernorm_forward_affine(M=8, N=1024, eps=1e-5):
    print("\n=== LayerNorm FWD (with affine) Capture ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    gamma = cp.random.randn(N).astype(cp.float32)
    beta  = cp.random.randn(N).astype(cp.float32)
    Y = cp.empty_like(X)

    def body():
        ln_ops.forward(X, gamma=gamma, beta=beta, eps=eps, out=Y)

    try_capture(body, "LayerNorm-FWD(affine)")


def test_layernorm_backward_no_affine(M=8, N=1024, eps=1e-5):
    print("\n=== LayerNorm BWD (no affine) Capture ===")
    X  = cp.random.randn(M, N).astype(cp.float32)
    # fwd 생성 (gamma/beta 없음)
    Y  = ln_ops.forward(X, gamma=None, beta=None, eps=eps)
    dY = cp.random.randn(M, N).astype(cp.float32)

    dX = cp.empty_like(X)

    def body():
        dx_out, dgamma, dbeta = ln_ops.backward(
            X, dY, gamma=None, eps=eps,
            out_dx=dX, out_dgamma=None, out_dbeta=None, return_param_grads=False
        )
        # dgamma/dbeta는 None이어야 함
        assert dgamma is None and dbeta is None

    try_capture(body, "LayerNorm-BWD(no-affine)")


def test_layernorm_backward_affine(M=8, N=1024, eps=1e-5):
    print("\n=== LayerNorm BWD (with affine) Capture ===")
    X  = cp.random.randn(M, N).astype(cp.float32)
    gamma = cp.random.randn(N).astype(cp.float32)
    beta  = cp.random.randn(N).astype(cp.float32)

    # fwd 생성 (affine)
    Y  = ln_ops.forward(X, gamma=gamma, beta=beta, eps=eps)
    dY = cp.random.randn(M, N).astype(cp.float32)

    dX = cp.empty_like(X)
    dgamma = cp.empty((N,), dtype=cp.float32)
    dbeta  = cp.empty((N,), dtype=cp.float32)

    def body():
        dx_out, dgamma_out, dbeta_out = ln_ops.backward(
            X, dY, gamma=gamma, eps=eps,
            out_dx=dX, out_dgamma=dgamma, out_dbeta=dbeta, return_param_grads=False
        )
        # 버퍼 전달했으므로 동일 포인터여야 함
        assert dx_out.data.ptr == dX.data.ptr
        assert dgamma_out is dgamma
        assert dbeta_out  is dbeta

    try_capture(body, "LayerNorm-BWD(affine)")


def test_layernorm_backward_affine_auto_grads(M=8, N=1024, eps=1e-5):
    print("\n=== LayerNorm BWD (with affine, auto param grads) Capture ===")
    X  = cp.random.randn(M, N).astype(cp.float32)
    gamma = cp.random.randn(N).astype(cp.float32)
    beta  = cp.random.randn(N).astype(cp.float32)

    Y  = ln_ops.forward(X, gamma=gamma, beta=beta, eps=eps)
    dY = cp.random.randn(M, N).astype(cp.float32)
    dX = cp.empty_like(X)

    # 자동 할당 경로
    def body():
        dx_out, dgamma_out, dbeta_out = ln_ops.backward(
            X, dY, gamma=gamma, eps=eps,
            out_dx=dX, out_dgamma=None, out_dbeta=None, return_param_grads=True
        )
        # 자동 할당된 반환값이 있어야 함
        assert dgamma_out is not None and dbeta_out is not None

    try_capture(body, "LayerNorm-BWD(affine-auto-grads)")


# ======================== main =========================
if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    # Forward
    test_layernorm_forward_no_affine(M=8, N=1024, eps=1e-5)
    test_layernorm_forward_affine(M=8, N=1024, eps=1e-5)

    # Backward
    test_layernorm_backward_no_affine(M=8, N=1024, eps=1e-5)
    test_layernorm_backward_affine(M=8, N=1024, eps=1e-5)
    test_layernorm_backward_affine_auto_grads(M=8, N=1024, eps=1e-5)

    print("\n[OK] layernorm capture tests finished.")
