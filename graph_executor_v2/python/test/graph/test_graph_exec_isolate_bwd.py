# python/test/graph/test_graph_exec_isolate_bwd.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import conv2d as conv_ops
from graph_executor_v2.ops import gemm  as gemm_ops

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
    """desc 설명과 함께 fn()을 캡쳐해 본다."""
    s = cp.cuda.Stream(non_blocking=True)
    # 워밍업 한 번 (플랜/핸들 생성 등 캡쳐 밖에서 끝내기)
    try:
        fn()
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return

    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
        print(f"[{desc}] capture: OK (type={type(g).__name__})")
        return g
    except Exception:
        print(f"[{desc}] capture: FAILED")
        traceback.print_exc()
        return None

def test_gemm_bwd_only(M=4, K=128, N=10, with_bias=True, act="none"):
    """GEMM(Linear) 역전파만 단독 캡쳐."""
    print("\n=== GEMM Backward-only Capture ===")
    A  = cp.random.randn(M, K).astype(cp.float32)
    W  = cp.random.randn(K, N).astype(cp.float32)
    Z  = cp.random.randn(M, N).astype(cp.float32)   # pre-activation
    gY = cp.random.randn(M, N).astype(cp.float32)

    gA = cp.empty_like(A)
    gB = cp.empty_like(W)
    gBias = cp.empty((1, N), dtype=cp.float32) if with_bias else None

    # 캡쳐-세이프 워크스페이스 (미리 할당)
    dZ_ws = cp.empty_like(Z)
    lt_ws = cp.empty(8 * 1024 * 1024, dtype=cp.uint8)

    def body():
        gemm_ops.backward_into(
            A, W, gY, Z,
            act=act, with_bias=with_bias,
            gA_out=gA, gB_out=gB, gBias_out=gBias,
            work_dZ=dZ_ws, lt_workspace=lt_ws
        )

    # 핸들/플랜 워밍업 이슈가 있으면 밖에서 한 번 더 호출
    body()
    cp.cuda.get_current_stream().synchronize()

    try_capture(body, "GEMM-BWD")

def test_conv_bwd_only(N=4, Cin=3, H=32, W=32, Cout=8, k=3, act="none", with_bias=True):
    """Conv2D 역전파만 단독 캡쳐."""
    print("\n=== Conv2D Backward-only Capture ===")
    X = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    KH = KW = k
    pad = (1, 1)
    stride = (1, 1)
    dil = (1, 1)
    groups = 1

    # conv weight: (Cout, Cin/groups, KH, KW)
    Wt = cp.random.randn(Cout, Cin // groups, KH, KW).astype(cp.float32)
    B  = cp.random.randn(Cout,).astype(cp.float32) if with_bias else None

    # fwd로 Y/Z 생성
    # WS(fwd)
    Ho = (H + 2*pad[0] - dil[0]*(KH-1) - 1)//stride[0] + 1
    Wo = (W + 2*pad[1] - dil[1]*(KW-1) - 1)//stride[1] + 1
    Y = cp.empty((N, Cout, Ho, Wo), dtype=cp.float32)

    ws_f = conv_ops.Conv2DWorkspaces()
    Kcol = (Cin // groups) * KH * KW
    HWo  = Ho * Wo
    ws_f.dCol   = cp.empty((HWo, Kcol),    dtype=cp.float32)
    ws_f.W_KC   = cp.empty((Kcol, Cout),   dtype=cp.float32)
    ws_f.Y_tmp  = cp.empty((HWo, Cout),    dtype=cp.float32)
    ws_f.Z_rows = cp.empty((HWo, Cout),    dtype=cp.float32)  # save_z

    conv_ops.forward_into(
        X, Wt, out=Y, B=B,
        stride=stride, padding=pad, dilation=dil, groups=groups,
        with_bias=with_bias, act="none",
        save_z=True, Z_saved=Y,   # act='none' -> Z==Y alias
        work=ws_f
    )

    gY = cp.random.randn(*Y.shape).astype(cp.float32)
    gX = cp.empty_like(X)
    gW = cp.empty_like(Wt)
    gB = cp.empty_like(B) if with_bias else None

    # WS(bwd)
    ws_b = conv_ops.Conv2DWorkspaces()
    ws_b.dCol_b  = cp.empty((HWo, Kcol), dtype=cp.float32)
    ws_b.dTmp    = cp.empty((max(Cout*Kcol, HWo*Kcol),), dtype=cp.float32)
    ws_b.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
    ws_b.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
    ws_b.W_CK    = cp.empty((Cout, Kcol), dtype=cp.float32)
    ws_b.dY_HT   = cp.empty((HWo,  Cout), dtype=cp.float32)
    ws_b.dWpack  = cp.empty((Cout, Kcol), dtype=cp.float32)

    def body():
        conv_ops.backward_into(
            X, Wt, gY, Y,
            stride=stride, padding=pad, dilation=dil, groups=groups,
            with_bias=with_bias, act=act,
            gX_out=gX, gW_out=gW, gB_out=gB,
            work=ws_b,
        )

    # 워밍업
    body()
    cp.cuda.get_current_stream().synchronize()

    try_capture(body, "Conv2D-BWD")

if __name__ == "__main__":
    cp.random.seed(1234)
    print("[ENV]", env_info())

    # 1) Linear(GEMM) 역전파만 캡쳐 시도
    test_gemm_bwd_only(M=4, K=3*32*32, N=10, with_bias=True, act="none")

    # 2) Conv2D 역전파만 캡쳐 시도
    test_conv_bwd_only(N=4, Cin=3, H=32, W=32, Cout=8, k=3, act="none", with_bias=True)

    print("\n[OK] isolate_bwd test finished.")
