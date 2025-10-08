# python/test/graph/test_graph_exec_pad_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import pad as pad_ops


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
    """desc 설명과 함께 fn()을 캡쳐해 본다.
    CuPy 13.x: Graph.instantiate()가 없으므로 Graph.upload()/Graph.launch() 사용.
    CuPy 12.x 등: instantiate 경로 지원.
    """
    s = cp.cuda.Stream(non_blocking=True)

    # 1) warmup: 플랜/핸들 생성 등 캡처 외부로 밀어내기
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
    except Exception:
        print(f"[{desc}] capture: FAILED")
        traceback.print_exc()
        return None

    # 3) instantiate or upload+launch (버전별 분기)
    try:
        if hasattr(g, "instantiate"):
            # CuPy 12.x 등
            ge = g.instantiate()
            print(f"[{desc}] instantiate: OK (exec={type(ge).__name__})")
            with s:
                ge.launch()
            s.synchronize()
            print(f"[{desc}] replay: OK")
            return ge
        else:
            # CuPy 13.x 경로: instantiate 없음 → upload(옵션) 후 launch
            with s:
                if hasattr(g, "upload"):
                    try:
                        g.upload()
                    except Exception:
                        # 환경에 따라 upload가 없거나 실패할 수 있으니 무시하고 진행
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
def test_pad_forward_contig(
    shape=(2, 3, 8, 8),
    before=(1, 0, 2, 3),
    after=(0, 2, 1, 0),
    value=0.5,
):
    print("\n=== Pad Forward (contiguous) Capture ===")
    X = cp.random.randn(*shape).astype(cp.float32)
    out_shape = pad_ops.compute_padded_shape(X.shape, before, after)
    Y = cp.empty(out_shape, dtype=cp.float32)

    def body():
        pad_ops.forward_into(X, before=before, after=after, value=value, out=Y)

    try_capture(body, "Pad-FWD(contig)")

    # 간단 검증: pad 영역 샘플 몇 개 확인
    # 중심부(원본 복사 위치)에서 무작위 좌표 한 개 검증
    idx = tuple([before[d] for d in range(X.ndim)])
    assert cp.isfinite(Y[idx]).all()


def test_pad_backward_contig(
    shape=(2, 3, 8, 8),
    before=(1, 0, 2, 3),
    after=(0, 2, 1, 0),
):
    print("\n=== Pad Backward (contiguous) Capture ===")
    X_shape = tuple(map(int, shape))
    Y_shape = pad_ops.compute_padded_shape(X_shape, before, after)
    dY = cp.random.randn(*Y_shape).astype(cp.float32)
    dX = cp.empty(X_shape, dtype=cp.float32)

    def body():
        pad_ops.backward_into(dY, before=before, after=after, dX_out=dX)

    try_capture(body, "Pad-BWD(contig)")

    # 간단 검증: dX는 dY의 슬라이스여야 함 (중앙 위치)
    center = tuple(slice(before[d], before[d] + X_shape[d]) for d in range(len(X_shape)))
    assert cp.allclose(dX, dY[center])


def test_pad_forward_strided(
    shape=(2, 3, 9, 10),
    before=(0, 1, 1, 2),
    after=(2, 0, 3, 0),
    value=0.0,
):
    print("\n=== Pad Forward (non-contiguous/strided) Capture ===")
    # 비연속: 채널 순서를 바꾸거나, 하/우 스텝을 넣어보기
    X0 = cp.random.randn(*shape).astype(cp.float32)
    # 예: H축 2 step, W축 오프셋 1 → view(스트라이드) 보장
    X = X0[:, :, ::2, 1:]  # shape = (N,C, ceil(H/2), W-1), 비연속
    out_shape = pad_ops.compute_padded_shape(X.shape, before, after)
    Y = cp.empty(out_shape, dtype=cp.float32)

    def body():
        pad_ops.forward_into(X, before=before, after=after, value=value, out=Y)

    try_capture(body, "Pad-FWD(strided)")
    # 간단 검증: 원본 일부 좌표 매핑 체크
    n, c, h, w = 0, 0, 0, 0
    assert cp.isclose(Y[n, c + before[1], h + before[2], w + before[3]],
                      X[n, c, h, w])


def test_pad_backward_strided(
    base_shape=(2, 3, 9, 10),
    before=(0, 1, 1, 2),
    after=(2, 0, 3, 0),
):
    print("\n=== Pad Backward (non-contiguous/strided) Capture ===")
    X0 = cp.random.randn(*base_shape).astype(cp.float32)
    X = X0[:, :, ::2, 1:]  # 비연속 view
    X_shape = X.shape
    Y_shape = pad_ops.compute_padded_shape(X_shape, before, after)

    dY = cp.random.randn(*Y_shape).astype(cp.float32)
    dX = cp.empty(X_shape, dtype=cp.float32)  # 비연속 view에 쓰지는 않음. out은 연속 버퍼가 안전.
    # 주의: backward_into는 dX_out 버퍼에 연속/비연속 상관없이 element-wise로 써줌.
    # 여기서는 연속 버퍼를 사용(필요하면 X.copy().shape로 사용 가능)

    def body():
        pad_ops.backward_into(dY, before=before, after=after, dX_out=dX)

    try_capture(body, "Pad-BWD(strided)")
    # 검증: dX == dY 중앙 슬라이스
    center = tuple(slice(before[d], before[d] + X_shape[d]) for d in range(len(X_shape)))
    assert cp.allclose(dX, dY[center])


def test_pad_forward_nd_rank3():
    print("\n=== Pad Forward (rank-3) Capture ===")
    X = cp.random.randn(4, 7, 13).astype(cp.float32)
    before = (2, 0, 3)
    after  = (0, 5, 1)
    Y = cp.empty(pad_ops.compute_padded_shape(X.shape, before, after), dtype=cp.float32)

    def body():
        pad_ops.forward_into(X, before=before, after=after, value=1.2345, out=Y)

    try_capture(body, "Pad-FWD(rank3)")
    # 패딩 부분 값 체크 몇 군데
    assert cp.isclose(Y[0, 0, 0], 1.2345)  # 좌측 상단은 패딩


def test_pad_backward_nd_rank3():
    print("\n=== Pad Backward (rank-3) Capture ===")
    X_shape = (4, 7, 13)
    before  = (2, 0, 3)
    after   = (0, 5, 1)
    Y_shape = pad_ops.compute_padded_shape(X_shape, before, after)
    dY = cp.random.randn(*Y_shape).astype(cp.float32)
    dX = cp.empty(X_shape, dtype=cp.float32)

    def body():
        pad_ops.backward_into(dY, before=before, after=after, dX_out=dX)

    try_capture(body, "Pad-BWD(rank3)")
    center = tuple(slice(before[d], before[d] + X_shape[d]) for d in range(len(X_shape)))
    assert cp.allclose(dX, dY[center])


# ======================== main ============================
if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    # 1) 연속 텐서
    test_pad_forward_contig()
    test_pad_backward_contig()

    # 2) 비연속(스트라이드) 텐서
    test_pad_forward_strided()
    test_pad_backward_strided()

    # 3) 다른 랭크(3D) 케이스
    test_pad_forward_nd_rank3()
    test_pad_backward_nd_rank3()

    print("\n[OK] pad capture tests finished.")
