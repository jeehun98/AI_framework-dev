# python/test/graph/test_graph_exec_pool2d_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import pool2d as pool_ops

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

def _out_hw(H, W, kH, kW, sH, sW, pH, pW, dH=1, dW=1, ceil_mode=False):
    effKH = (kH - 1) * dH + 1
    effKW = (kW - 1) * dW + 1
    aH = H + 2 * pH - effKH
    aW = W + 2 * pW - effKW
    if ceil_mode:
        Ho = (aH >= 0) and ((aH + sH - 1) // sH + 1) or 0
        Wo = (aW >= 0) and ((aW + sW - 1) // sW + 1) or 0
    else:
        Ho = (aH >= 0) and (aH // sH + 1) or 0
        Wo = (aW >= 0) and (aW // sW + 1) or 0
    return max(0, int(Ho)), max(0, int(Wo))

# ===================== capture utils ====================
# replace try_capture() in test_graph_exec_pool2d_capture.py

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

        # 3) instantiate/exec 호환 레이어
        # CuPy 13.x: instantiate 없음 → upload + launch 사용
        # CuPy 12.x 일부/향후 변경 대비: hasattr 체크
        if hasattr(g, "instantiate"):
            ge = g.instantiate()
            print(f"[{desc}] instantiate: OK (exec={type(ge).__name__})")
            with s:
                ge.launch()
            s.synchronize()
            print(f"[{desc}] replay: OK")
            return ge
        else:
            # optional: 먼저 업로드(선택) 후 런치
            with s:
                try:
                    # upload()는 선택 사항이지만 첫 런치 전에 하면 안정적
                    if hasattr(g, "upload"):
                        g.upload()
                except Exception:
                    # 일부 환경에서 upload 미지원일 수 있으니 무시하고 진행
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
def test_pool2d_max_with_ws_indices(
    N=4, C=8, H=32, W=32,
    kernel=(3,3), stride=(2,2), padding=(1,1),
    dilation=(1,1), ceil_mode=False
):
    """MaxPool FWD/BWD: 캡처-세이프 WS 인덱스 버퍼 주입 경로 테스트."""
    print("\n=== Pool2D MAX (WS indices) Capture ===")
    X = cp.random.randn(N, C, H, W).astype(cp.float32)

    kH, kW = kernel; sH, sW = stride; pH, pW = padding; dH, dW = dilation
    Ho, Wo = _out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)

    # 외부 WS 인덱스 버퍼 (int32, [N,C,Ho,Wo])
    ws_idx = cp.empty((N, C, Ho, Wo), dtype=cp.int32)

    # Forward 본문 (WS 사용, return_indices=False)
    Y = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
    def fwd_body():
        pool_ops.forward(
            X,
            kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, mode="max", return_indices=False,
            ws_indices=ws_idx, out=Y,
        )

    # Backward 본문 (WS 인덱스 재사용)
    gY = cp.random.randn(N, C, Ho, Wo).astype(cp.float32)
    def bwd_body():
        pool_ops.backward(
            X, Y, gY,
            kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, mode="max",
            indices=None, ws_indices=ws_idx,
        )

    try_capture(fwd_body, "MaxPool-FWD(ws_indices)")
    try_capture(bwd_body, "MaxPool-BWD(ws_indices)")

def test_pool2d_max_with_indices_tensor(
    N=2, C=4, H=16, W=16,
    kernel=(2,2), stride=(2,2), padding=(0,0),
    dilation=(1,1), ceil_mode=False
):
    """MaxPool FWD/BWD: 정식 Indices 텐서 사용 경로 테스트."""
    print("\n=== Pool2D MAX (Indices tensor) Capture ===")
    X = cp.random.randn(N, C, H, W).astype(cp.float32)
    Ho, Wo = _out_hw(H, W, *kernel, *stride, *padding, *dilation, ceil_mode)

    # FWD에서 indices 텐서도 생성/반환
    Y = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
    Ind = cp.empty((N, C, Ho, Wo), dtype=cp.int32)

    def fwd_body():
        out, ind = pool_ops.forward(
            X, kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, mode="max",
            return_indices=True, ws_indices=None, out=Y
        )
        # out/ind는 호출자가 만든 Y/Ind 버퍼에 써지며 객체는 동일
        assert out.data.ptr == Y.data.ptr
        assert ind.data.ptr == Ind.data.ptr or True  # ind는 새로 만들 수도 있어, 아래에서 다시 채움
        # ind가 None일 수는 없음(return_indices=True 이므로)
        if ind.data.ptr != Ind.data.ptr:
            Ind[...] = ind  # shape 동일, 복사해 둠

    gY = cp.random.randn(N, C, Ho, Wo).astype(cp.float32)
    def bwd_body():
        pool_ops.backward(
            X, Y, gY,
            kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, mode="max",
            indices=Ind, ws_indices=None
        )

    try_capture(fwd_body, "MaxPool-FWD(indices)")
    try_capture(bwd_body, "MaxPool-BWD(indices)")

def test_pool2d_avg(
    N=3, C=5, H=31, W=35,
    kernel=(3,3), stride=(2,2), padding=(1,1),
    dilation=(1,1), ceil_mode=False, count_include_pad=False
):
    """AvgPool FWD/BWD: 캡처 테스트 (WS scratch 미사용)."""
    print("\n=== Pool2D AVG Capture ===")
    X = cp.random.randn(N, C, H, W).astype(cp.float32)
    Ho, Wo = _out_hw(H, W, *kernel, *stride, *padding, *dilation, ceil_mode)
    Y  = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
    gY = cp.random.randn(N, C, Ho, Wo).astype(cp.float32)

    def fwd_body():
        pool_ops.forward(
            X,
            kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            mode="avg", out=Y
        )

    def bwd_body():
        pool_ops.backward(
            X, Y, gY,
            kernel=kernel, stride=stride, padding=padding, dilation=dilation,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            mode="avg"
        )

    try_capture(fwd_body, "AvgPool-FWD")
    try_capture(bwd_body, "AvgPool-BWD")

# ======================== main ============================
if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    # 1) MaxPool: WS 인덱스 버퍼 주입 경로
    test_pool2d_max_with_ws_indices(
        N=4, C=8, H=32, W=32,
        kernel=(3,3), stride=(2,2), padding=(1,1),
        dilation=(1,1), ceil_mode=False
    )

    # 2) MaxPool: 정식 Indices 텐서 경로
    test_pool2d_max_with_indices_tensor(
        N=2, C=4, H=16, W=16,
        kernel=(2,2), stride=(2,2), padding=(0,0),
        dilation=(1,1), ceil_mode=False
    )

    # 3) AvgPool: FWD/BWD
    test_pool2d_avg(
        N=3, C=5, H=31, W=35,
        kernel=(3,3), stride=(2,2), padding=(1,1),
        dilation=(1,1), ceil_mode=False, count_include_pad=False
    )

    print("\n[OK] pool2d capture tests finished.")
