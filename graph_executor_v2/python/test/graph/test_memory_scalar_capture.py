# python/test/ops/test_memory_scalar_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import memory as mem


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
    # warmup
    try:
        fn(s); cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED"); traceback.print_exc(); return None
    # capture & replay
    try:
        with s:
            s.begin_capture()
            fn(s)
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
        return g, s
    except Exception:
        print(f"[{desc}] capture/replay: FAILED"); traceback.print_exc(); return None, None


def test_fill_f32_capture(shape=(1024,), value=3.25):
    print("\n=== memory.fill_f32 Capture ===")
    a = cp.empty(shape, dtype=cp.float32)

    def body(stream):
        mem.fill_f32(a, value, stream=stream.ptr)  # 고정 인자 → 캡처-세이프

    g, s = try_capture(body, "fill_f32")
    assert g is not None

    # 검증: 캡처된 값으로 채워졌는지
    mx = float(cp.max(cp.abs(a - value)))
    print(f"[fill_f32] max|a - {value}| = {mx:.3e}")
    assert mx == 0.0

    # 캡처 밖에서 값 변경 테스트(그래프와 무관하게 동작해야 함)
    mem.fill_f32(a, 1.0, stream=None)
    mx2 = float(cp.max(cp.abs(a - 1.0)))
    print(f"[fill_f32 outside] max|a - 1.0| = {mx2:.3e}")
    assert mx2 == 0.0


def test_fill_i32_capture(shape=(257,), value=7):
    print("\n=== memory.fill_i32 Capture ===")
    a = cp.empty(shape, dtype=cp.int32)

    def body(stream):
        mem.fill_i32(a, value, stream=stream.ptr)  # 고정 인자 → 캡처-세이프

    g, s = try_capture(body, "fill_i32")
    assert g is not None

    # 검증
    ok = bool(int(cp.max(cp.abs(a - value))) == 0)
    print(f"[fill_i32] all_equal={ok}")
    assert ok

    # 캡처 밖 변경 테스트
    mem.fill_i32(a, -3, stream=None)
    ok2 = bool(int(cp.max(cp.abs(a + 3))) == 0)
    print(f"[fill_i32 outside] all_equal={ok2}")
    assert ok2


if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())
    test_fill_f32_capture(shape=(4096,), value=2.5)
    test_fill_i32_capture(shape=(1023,), value=11)
    print("\n[OK] memory scalar capture tests finished.")
