# python/test/graph/test_graph_exec_softmax_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import softmax as smx


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
    """desc 설명과 함께 fn()을 캡처하여 그래프 실행까지 검증.
    CuPy 13.x: Graph.instantiate() 없음 → upload()/launch() 사용
    CuPy 12.x: instantiate() 경로 사용
    """
    s = cp.cuda.Stream(non_blocking=True)

    # warmup (플랜/핸들 등 캡처 외부에서 완료)
    try:
        fn()
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return None

    # capture + replay
    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
        print(f"[{desc}] capture: OK (graph={type(g).__name__})")

        if hasattr(g, "instantiate"):
            ge = g.instantiate()
            print(f"[{desc}] instantiate: OK (exec={type(ge).__name__})")
            with s:
                ge.launch()
            s.synchronize()
            print(f"[{desc}] replay: OK")
            return ge
        else:
            with s:
                if hasattr(g, "upload"):
                    try:
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


# ===================== math reference ====================
def _apply_mask(x: cp.ndarray, mask: cp.ndarray | None):
    if mask is None:
        return x
    if mask.ndim == 2:
        M, N = x.shape
        if mask.shape == (M, N):
            return x + mask
        if mask.shape == (1, N):
            return x + mask
        if mask.shape == (M, 1):
            return x + mask
        raise ValueError("unsupported mask shape")
    elif mask.ndim == 1:
        # [N] → [1,N]
        return x + mask[cp.newaxis, :]
    else:
        raise ValueError("unsupported mask ndim")

def ref_softmax_forward(x, mask=None, scale=1.0, log=False):
    z = _apply_mask(x, mask) * scale
    # 안정화
    m = z.max(axis=1, keepdims=True)
    ez = cp.exp(z - m)
    denom = ez.sum(axis=1, keepdims=True)
    if not log:
        return ez / denom
    else:
        return cp.log(ez) - cp.log(denom)

def ref_softmax_backward(y_or_logy, gy, scale=1.0, log=False):
    if not log:
        # y: softmax
        dot = (gy * y_or_logy).sum(axis=1, keepdims=True)
        return scale * (gy - dot) * y_or_logy
    else:
        # y_or_logy: log-softmax → p = exp(y)
        p = cp.exp(y_or_logy)
        sum_dy = gy.sum(axis=1, keepdims=True)
        return scale * (gy - sum_dy * p)


# ======================== tests =========================
def test_softmax_forward_no_mask(M=8, N=17, scale=0.7):
    print("\n=== Softmax FWD (no mask) Capture+Numerics ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    Y = cp.empty_like(X)

    def body():
        smx.forward(X, scale=scale, log=False, out=Y)

    try_capture(body, "Softmax-FWD(no-mask)")

    Y_ref = ref_softmax_forward(X, None, scale=scale, log=False)
    assert cp.allclose(Y, Y_ref, rtol=1e-5, atol=1e-6)


def test_logsoftmax_forward_with_masks(M=7, N=19, scale=1.25):
    print("\n=== LogSoftmax FWD (mask broadcast variants) Capture+Numerics ===")
    X = cp.random.randn(M, N).astype(cp.float32)

    masks = {
        "[M,N]": cp.random.randn(M, N).astype(cp.float32),
        "[1,N]": cp.random.randn(1, N).astype(cp.float32),
        "[M,1]": cp.random.randn(M, 1).astype(cp.float32),
        "[N]  ": cp.random.randn(N).astype(cp.float32),
    }

    for label, mask in masks.items():
        print(f"  - case {label}")
        Y = cp.empty_like(X)

        def body_fwd():
            smx.forward(X, mask=mask, scale=scale, log=True, out=Y)

        try_capture(body_fwd, f"LogSoftmax-FWD(mask={label})")

        Y_ref = ref_softmax_forward(X, mask, scale=scale, log=True)
        assert cp.allclose(Y, Y_ref, rtol=1e-5, atol=1e-6)


def test_softmax_backward_y_provided(M=6, N=23, scale=0.5):
    print("\n=== Softmax BWD (y_provided=True) Capture+Numerics ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    # forward (no mask)
    Y = smx.forward(X, scale=scale, log=False)

    gY = cp.random.randn(M, N).astype(cp.float32)
    dX = cp.empty_like(X)

    def body_bwd():
        smx.backward(Y, gY, scale=scale, log=False, y_provided=True, out=dX)

    try_capture(body_bwd, "Softmax-BWD(y_provided=True)")

    dX_ref = ref_softmax_backward(Y, gY, scale=scale, log=False)
    assert cp.allclose(dX, dX_ref, rtol=1e-5, atol=1e-6)


def test_logsoftmax_backward_y_provided_with_mask(M=5, N=21, scale=0.9):
    print("\n=== LogSoftmax BWD (y_provided=True, with mask) Capture+Numerics ===")
    X = cp.random.randn(M, N).astype(cp.float32)
    mask = cp.random.randn(1, N).astype(cp.float32)  # 브로드캐스트 케이스 예시

    # forward(log=True, with mask)
    Ylog = smx.forward(X, mask=mask, scale=scale, log=True)

    gY = cp.random.randn(M, N).astype(cp.float32)
    dX = cp.empty_like(X)

    def body_bwd():
        smx.backward(Ylog, gY, scale=scale, log=True, y_provided=True, out=dX)

    try_capture(body_bwd, "LogSoftmax-BWD(y_provided=True, mask used in fwd)")

    dX_ref = ref_softmax_backward(Ylog, gY, scale=scale, log=True)
    assert cp.allclose(dX, dX_ref, rtol=1e-5, atol=1e-6)


# ======================== main ============================
if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    test_softmax_forward_no_mask()
    test_logsoftmax_forward_with_masks()
    test_softmax_backward_y_provided()
    test_logsoftmax_backward_y_provided_with_mask()

    print("\n[OK] softmax capture tests finished.")
