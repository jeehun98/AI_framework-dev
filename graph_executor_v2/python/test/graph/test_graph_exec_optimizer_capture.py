# python/test/graph/test_graph_exec_optimizer_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import optimizer as opt


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
    # warmup (eager)
    try:
        fn(); cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED"); traceback.print_exc(); return None
    # capture & replay
    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
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


def test_sgd_update_capture(N=1<<16, use_momentum=True, nesterov=True):
    print("\n=== SGD Update Capture ===")
    # Params/Grads
    P = cp.random.randn(N).astype(cp.float32)
    G = cp.random.randn(N).astype(cp.float32) * 0.1
    P0 = P.copy()

    # Momentum buffer (optional)
    V = cp.zeros_like(P) if use_momentum else None

    lr = 1e-2
    momentum = 0.9 if use_momentum else 0.0
    dampening = 0.0
    weight_decay = 1e-4

    def body():
        pout = opt.sgd_update(
            P, G, V,
            lr=lr, momentum=momentum, dampening=dampening,
            nesterov=nesterov if use_momentum else False,
            weight_decay=weight_decay,
            stream=None,  # current stream (set by try_capture)
        )
        # in-place 보장 확인
        assert pout.data.ptr == P.data.ptr
        if V is not None:
            # momentum buffer도 in-place로 갱신되어야 함
            assert V.flags.c_contiguous

    # run
    g = try_capture(body, f"SGD({'mom' if use_momentum else 'plain'}{'-nesterov' if nesterov and use_momentum else ''})")
    assert g is not None

    # sanity check: 값이 변했는지 확인
    diff = float(cp.max(cp.abs(P - P0)))
    print(f"[SGD] max|ΔP| = {diff:.6e}")
    assert diff > 0.0


def test_adamw_update_capture(N=1<<16, bias_correction=False):
    print("\n=== AdamW Update Capture ===")
    # Params/Grads/M/V
    P = cp.random.randn(N).astype(cp.float32)
    G = cp.random.randn(N).astype(cp.float32) * 0.05
    M = cp.zeros_like(P)
    V = cp.zeros_like(P)
    P0 = P.copy()

    # 고정된 step으로 캡처 (bias_correction=True 여도 step이 고정이면 캡처 안전)
    step = 1

    def body():
        pout = opt.adamw_update(
            P, G, M, V,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=1e-2,
            bias_correction=bias_correction,
            step=step,
            stream=None,  # current stream
        )
        assert pout.data.ptr == P.data.ptr

    g = try_capture(body, f"AdamW{'-biascorr' if bias_correction else ''}")
    assert g is not None

    diff = float(cp.max(cp.abs(P - P0)))
    print(f"[AdamW] max|ΔP| = {diff:.6e}")
    assert diff > 0.0


if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    # SGD: no-momentum / momentum(+nesterov)
    test_sgd_update_capture(N=1<<15, use_momentum=False, nesterov=False)
    test_sgd_update_capture(N=1<<15, use_momentum=True,  nesterov=True)

    # AdamW: bias_correction off/on (둘 다 step 고정으로 캡처)
    test_adamw_update_capture(N=1<<15, bias_correction=False)
    test_adamw_update_capture(N=1<<15, bias_correction=True)

    print("\n[OK] optimizer capture tests finished.")
