# File: python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional, Sequence
import cupy as cp
from .capture_plan import CapturePlan

# Conv2D / WS 유틸
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.ops import conv2d as convops

# (선택) BN2d 타입 감지용
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d
except Exception:
    _BN2d = None

# ===== NVTX (optional) =====
# - 타임라인 분석을 위해 범위를 좁게 잘라 태깅
try:
    from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range  # type: ignore
except Exception:
    class _DummyNvtx:
        def __call__(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            return _Ctx()
    nvtx_range = _DummyNvtx()  # type: ignore


class GraphExecLike:
    """CUDA Graph 미사용(또는 불가) 환경에서의 폴백 실행자.

    - 인터페이스를 graphExec(instantiated graph)와 최대한 동일하게 맞춘다.
    - 내부에 한 스텝을 수행하는 람다/클로저(_launch)를 보관하고,
      .launch(stream_ptr)로 호출되도록 한다.
    """
    def __init__(self, launch_fn, stream: cp.cuda.Stream):
        self._launch = launch_fn
        self._stream = stream

    def launch(self, stream_ptr=None):
        # stream_ptr는 호환성용 인자(무시). 내부에서 고정 스트림 사용.
        with self._stream:
            self._launch()


# ---------------- shape / workspace helpers ----------------
def _out_hw(
    H: int, W: int, KH: int, KW: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Conv2D 출력 H/W 계산 (PyTorch 동일 공식을 정수 연산으로)."""
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    H_out = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
    return H_out, W_out


def _alloc_conv2d_ws(HWo: int, K: int, Cout: int) -> convops.Conv2DWorkspaces:
    """Conv2D용 워크스페이스 일괄 할당.

    - Forward/Backward 공용 버퍼를 모두 잡아두어 capture-safe(동적 malloc 회피).
    - dtype은 현재 커널 제약 상 float32 고정.
    """
    ws = convops.Conv2DWorkspaces()
    # Forward (옵션 A: 항상 Z_rows 필요)
    ws.dCol = cp.empty((HWo, K), dtype=cp.float32)
    ws.W_KC = cp.empty((K, Cout), dtype=cp.float32)
    ws.Y_tmp = cp.empty((HWo, Cout), dtype=cp.float32)
    ws.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)
    # Backward (공통)
    ws.dCol_b = cp.empty((HWo, K), dtype=cp.float32)
    ws.dTmp = cp.empty((max(Cout * K, HWo * K),), dtype=cp.float32)
    ws.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
    ws.Z_rows_b = cp.empty((Cout, HWo), dtype=cp.float32)
    # Backward 옵션 (gX, gW)
    ws.W_CK = cp.empty((Cout, K), dtype=cp.float32)
    ws.dY_HT = cp.empty((HWo, Cout), dtype=cp.float32)
    ws.dWpack = cp.empty((Cout, K), dtype=cp.float32)
    return ws


def _ensure_conv2d_ws_for_forward(
    per, lyr: Conv2D, cur_shape: tuple[int, int, int, int]
) -> convops.Conv2DWorkspaces:
    """Forward 직전에 Conv2D WS를 준비하여 반환.

    - plan.per_layer[i].work 가 있으면 재사용
    - 없으면 현재 입력 shape/레이어 설정으로 새로 계산/할당 후 per.work에 저장 시도
    """
    ws = getattr(per, "work", None)
    if ws is not None:
        return ws

    N, Cin, H, W = map(int, cur_shape)
    KH, KW = lyr.kernel_size
    Cout = int(lyr.out_channels)
    groups = int(lyr.groups)

    H_out, W_out = _out_hw(H, W, KH, KW, lyr.stride, lyr.padding, lyr.dilation)
    HWo = H_out * W_out
    K = (Cin // groups) * KH * KW

    ws = _alloc_conv2d_ws(HWo, K, Cout)

    # per가 동적 속성 할당을 지원하면 저장 (지원하지 않아도 무시)
    try:
        setattr(per, "work", ws)
    except Exception:
        pass

    return ws


def _ensure_conv2d_ws_for_backward(per, lyr: Conv2D) -> convops.Conv2DWorkspaces:
    """Backward 직전에 Conv2D WS를 준비하여 반환.

    - forward 시점에 per.work가 없었다면, per.y/lyr.W로부터 크기를 역추정해 생성.
    """
    ws = getattr(per, "work", None)
    if ws is not None:
        return ws

    # per.y: (N, Cout, H_out, W_out)
    N, Cout, H_out, W_out = map(int, per.y.shape)
    HWo = H_out * W_out

    # lyr.W: (Cout, Cin, KH, KW) (groups 고려는 K 계산에서 반영)
    _, Cin, KH, KW = map(int, lyr.W.shape)
    groups = int(lyr.groups)
    K = (Cin // groups) * KH * KW

    ws = _alloc_conv2d_ws(HWo, K, Cout)

    try:
        setattr(per, "work", ws)
    except Exception:
        pass

    return ws


# ---------------- grads zeroing ----------------
def _zero_bwd_buffers(plan: CapturePlan):
    """Backward 누적 방지: 캡처 내부에서 gA/gW/gB를 명시적으로 0세팅."""
    for p in plan.per_layer:
        if p.gA is not None:
            p.gA.fill(0)
        if p.gW is not None:
            p.gW.fill(0)
        if p.gB is not None:
            p.gB.fill(0)


# ---------------- forward / backward runners ----------------
def _run_fwd(model, plan: CapturePlan, X, stream_ptr: Optional[int], *, layers_override: Optional[Sequence[Any]] = None):
    """Forward 러너 (capture/replay 공용).

    - layers_override가 주어지면 그 시퀀스를 사용(동적 경로 전개 지원)
    - 아닐 경우 model.layers 사용(정적 경로)
    - 각 레이어는 가능한 한 capture-safe 시그니처 (out, z_out, work, stream)를 우선 시도
    """
    cur = X
    layers = list(layers_override) if layers_override is not None else list(model.layers)

    for i, lyr in enumerate(layers):
        per = plan.per_layer[i]
        ybuf = per.y
        zbuf = per.z

        ws_local = None
        if isinstance(lyr, Conv2D):
            ws_local = _ensure_conv2d_ws_for_forward(per, lyr, cur.shape)

        try:
            # 1st try: work + z_out
            lyr.forward_into(
                cur,
                out=ybuf,
                z_out=zbuf,
                work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                stream=stream_ptr,
            )
        except TypeError:
            # 2nd try: 최소 인자 (out, stream)만
            lyr.forward_into(cur, out=ybuf, stream=stream_ptr)

        cur = ybuf
    return cur


def _run_bwd(model, plan: CapturePlan, g_in, stream_ptr: Optional[int], *, layers_override: Optional[Sequence[Any]] = None):
    """Backward 러너 (capture/replay 공용).

    - BN2d 특수경로: 항상 X_saved(prev_y) 필요 (affine 여부 무관)
    - Conv2D는 backward 시에도 WS 보장이 필요하여 ensure 수행
    """
    layers = list(layers_override) if layers_override is not None else list(model.layers)

    for ridx, lyr in enumerate(reversed(layers)):
        i = len(layers) - 1 - ridx
        per = plan.per_layer[i]

        ws_local = None
        if isinstance(lyr, Conv2D):
            ws_local = _ensure_conv2d_ws_for_backward(per, lyr)

        # ---- BN2d: always needs X_saved regardless of affine ----
        is_bn = (_BN2d is not None and isinstance(lyr, _BN2d))
        if is_bn:
            prev_y = plan.per_layer[i - 1].y if i - 1 >= 0 else getattr(model, "_graph_input_buf", None)
            # gW_out/gB_out은 affine=False면 None이어도 OK
            lyr.backward_into(
                g_in,
                gA_out=per.gA,
                gW_out=per.gW,
                gB_out=per.gB,
                X_saved=prev_y,
                stream=stream_ptr,
            )
            g_in = per.gA
            continue  # BN 처리는 끝, 다음 레이어로

        # ---- 일반 경로 (Dense/Conv/기타) ----
        if per.gW is not None:
            # (가급적 work를 전달하되, 커널 시그니처 차이에 대비해 호환 분기)
            try:
                lyr.backward_into(
                    g_in,
                    gA_out=per.gA,
                    gW_out=per.gW,
                    gB_out=per.gB,
                    work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                    stream=stream_ptr,
                )
            except TypeError:
                try:
                    ws = ws_local if ws_local is not None else getattr(per, "work", None)
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA,
                        gW_out=per.gW,
                        gB_out=per.gB,
                        work_dZ=(getattr(ws, "dZ", None) if ws is not None else None),
                        lt_workspace=(getattr(ws, "lt_ws", None) if ws is not None else None),
                        stream=stream_ptr,
                    )
                except TypeError:
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA,
                        gW_out=per.gW,
                        gB_out=per.gB,
                        stream=stream_ptr,
                    )
        else:
            try:
                lyr.backward_into(
                    g_in,
                    gA_out=per.gA,
                    work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                    stream=stream_ptr,
                )
            except TypeError:
                lyr.backward_into(g_in, gA_out=per.gA, stream=stream_ptr)

        g_in = per.gA
    return g_in


# ---------------- record / instantiate ----------------
def record_step_graph(
    model,
    loss_fn,
    optimizer_step_fn,
    plan: CapturePlan,
    *,
    X_buf: cp.ndarray,
    y_buf: cp.ndarray,
    stream: Optional[cp.cuda.Stream] = None,
    loss_out: Optional[cp.ndarray] = None,  # ✅ 그래프 내부에서 갱신될 손실 스칼라 버퍼(디바이스, shape=())
    # ---- 확장 인자 (동적 경로/플래너) ----
    layers_override: Optional[Sequence[Any]] = None,  # 동적 경로 전개 결과(없으면 model.layers)
    exec_plan: Optional[Any] = None,                  # TODO: Execution Planner 결과 (스케줄/스트림 계획)
):
    """fwd → loss → bwd → opt '한 스텝'을 CUDA Graph로 캡처하여 실행자 반환.

    동작 개요:
      1) (워밍업 1회) 동일 순서로 한 번 실행하여 버퍼/워크스페이스/시그니처를 고정
         - loss_out이 주어졌다면 디바이스 스칼라를 여기에 기록
      2) CUDA Graph 캡처 (지원 시)
         - capture_stream(stream) 구간 안에서 동일 시퀀스를 수행
      3) instantiate() 하여 graphExec 반환
      4) CUDA Graph 미지원이면 GraphExecLike 폴백 반환

    확장 포인트:
      - layers_override: 동적 경로 전개(Sequential._linearize_path)의 레이어 시퀀스 지원
      - exec_plan: Execution Planner 결과(스트림/이벤트 스케줄 등)를 해석하여
        _run_fwd/_run_bwd 내부 호출 순서/스트림을 세분화할 수 있음 (현재는 보존호출)

    주의:
      - BN2d backward는 항상 X_saved가 필요하므로 forward 이전 출력(prev_y) 또는
        모델 입력 버퍼를 참조(첫 레이어가 BN인 경우를 대비).
      - 모든 메모리/워크스페이스는 capture-safe(사전 할당) 원칙을 따른다.
    """
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)

    dY = plan.loss.dY

    # ------ 워밍업 ------
    with nvtx_range("[CAPTURE] warmup"):
        with stream:
            # 그래프 입력 버퍼를 BN bwd fallback용으로 보관(필요시 참조)
            setattr(model, "_graph_input_buf", X_buf)

            cur = _run_fwd(model, plan, X_buf, stream.ptr, layers_override=layers_override)

            # loss forward: (loss_scalar_dev, dY_tmp)
            loss_dev, dY_tmp = loss_fn.forward(cur, y_buf)

            # 손실값을 고정 버퍼로 복사(있으면)
            if loss_out is not None:
                loss_out[...] = loss_dev

            # dY 고정/복사 (플랜의 dY shape과 맞으면 복사)
            g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
            if dY is not None:
                dY[...] = dY_tmp

            _zero_bwd_buffers(plan)
            _run_bwd(model, plan, g_in, stream.ptr, layers_override=layers_override)
            optimizer_step_fn()

    has_graph = hasattr(cp.cuda, "graph") and hasattr(cp.cuda.graph, "capture_stream")

    if has_graph:
        with nvtx_range("[CAPTURE] cudaGraphCapture"):
            with stream:
                with cp.cuda.graph.capture_stream(stream) as cap:
                    # (선택) exec_plan 해석하여 스트림/이벤트 스케줄 적용 가능
                    # TODO: GraphRuntime.run_step(exec_plan, ..., capture=True)로 이 블록을 대체
                    cur = _run_fwd(model, plan, X_buf, stream.ptr, layers_override=layers_override)

                    loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)

                    if loss_out is not None:
                        loss_out[...] = loss_dev

                    g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
                    if dY is not None:
                        dY[...] = dY_tmp

                    _zero_bwd_buffers(plan)
                    _run_bwd(model, plan, g_in, stream.ptr, layers_override=layers_override)
                    optimizer_step_fn()
        gexec = cap.graph.instantiate()
        return gexec

    # ------ 폴백 (그래프 미지원) ------
    def _one_step():
        cur = _run_fwd(model, plan, X_buf, stream.ptr, layers_override=layers_override)
        loss_dev, dY_tmp = loss_fn.forward(cur, y_buf)
        if loss_out is not None:
            loss_out[...] = loss_dev
        g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
        if dY is not None:
            dY[...] = dY_tmp
        _zero_bwd_buffers(plan)
        _run_bwd(model, plan, g_in, stream.ptr, layers_override=layers_override)
        optimizer_step_fn()

    return GraphExecLike(_one_step, stream)


class TrainGraph:
    """캡처된 그래프 실행자 + I/O 버퍼 묶음.

    - set_batch(): 호스트/다른 디바이스 텐서를 고정 I/O 버퍼로 복사
    - launch(): CUDA Graph 인스턴스(or 폴백)의 .launch 호출
    """
    def __init__(self, gexec, io, stream):
        self._gexec = gexec
        self._io = io
        self._stream = stream

    @property
    def logits(self):
        return self._io["logits"]

    @property
    def X_buf(self):
        return self._io["X"]

    @property
    def y_buf(self):
        return self._io["y"]

    def set_batch(self, X_dev, y_dev):
        """현재 배치를 고정 I/O 버퍼(X/y)에 복사 (그래프와 동일 스트림)."""
        xb, yb = self._io["X"], self._io["y"]
        with self._stream:  # ✅ 그래프와 동일 스트림에서 H2D/D2D 수행
            xb[...] = cp.asarray(X_dev, dtype=xb.dtype)
            yb[...] = cp.asarray(y_dev, dtype=yb.dtype)

    def launch(self):
        """CUDA Graph 인스턴스(or 폴백) 실행."""
        self._gexec.launch(self._stream.ptr)
