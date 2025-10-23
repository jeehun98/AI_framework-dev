# python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional
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


class GraphExecLike:
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
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    H_out = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
    return H_out, W_out


def _alloc_conv2d_ws(HWo: int, K: int, Cout: int) -> convops.Conv2DWorkspaces:
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
    """
    Forward 직전에 Conv2D WS를 준비하여 반환.
    per에 저장이 가능하면 저장도 시도하지만, 저장 실패해도 반환값을 즉시 사용.
    """
    # 이미 있다면 그대로 사용
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
    """
    Backward 직전에 Conv2D WS를 준비하여 반환.
    per.work가 없으면 per.y(=아웃풋)와 lyr.W로부터 크기를 재계산해 즉시 생성.
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
    for p in plan.per_layer:
        if p.gA is not None:
            p.gA.fill(0)
        if p.gW is not None:
            p.gW.fill(0)
        if p.gB is not None:
            p.gB.fill(0)


# ---------------- forward / backward runners ----------------
def _run_fwd(model, plan: CapturePlan, X, stream_ptr: Optional[int]):
    cur = X
    for i, lyr in enumerate(model.layers):
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


def _run_bwd(model, plan: CapturePlan, g_in, stream_ptr: Optional[int]):
    for ridx, lyr in enumerate(reversed(model.layers)):
        i = len(model.layers) - 1 - ridx
        per = plan.per_layer[i]

        ws_local = None
        if isinstance(lyr, Conv2D):
            ws_local = _ensure_conv2d_ws_for_backward(per, lyr)

        # ---- BN2d: always needs X_saved regardless of affine ----
        is_bn = (_BN2d is not None and isinstance(lyr, _BN2d))
        if is_bn:
            prev_y = plan.per_layer[i - 1].y if i - 1 >= 0 else None
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

        # ---- 기존 경로 (Dense/Conv/기타) ----
        if per.gW is not None:
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
):
    """
    fwd → loss → bwd → opt 한 스텝을 CUDA Graph로 녹화해 실행자 반환.
    Graph 미지원이면 Pseudo 실행자(GraphExecLike) 반환.
    """
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)

    dY = plan.loss.dY

    # ------ 워밍업 ------
    with stream:
        # 그래프 입력 버퍼를 BN bwd fallback용으로 보관(필요시 참조)
        setattr(model, "_graph_input_buf", X_buf)
        cur = _run_fwd(model, plan, X_buf, stream.ptr)
        loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)
        # ✅ 손실값을 고정 버퍼로 복사(있으면)
        if loss_out is not None:
            loss_out[...] = loss_dev
        g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
        if dY is not None:
            dY[...] = dY_tmp
        _zero_bwd_buffers(plan)
        _run_bwd(model, plan, g_in, stream.ptr)
        optimizer_step_fn()

    has_graph = hasattr(cp.cuda, "graph") and hasattr(cp.cuda.graph, "capture_stream")

    if has_graph:
        with stream:
            with cp.cuda.graph.capture_stream(stream) as cap:
                cur = _run_fwd(model, plan, X_buf, stream.ptr)
                loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)
                # ✅ 손실값을 고정 버퍼로 복사(있으면)
                if loss_out is not None:
                    loss_out[...] = loss_dev
                g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
                if dY is not None:
                    dY[...] = dY_tmp
                _zero_bwd_buffers(plan)
                _run_bwd(model, plan, g_in, stream.ptr)
                optimizer_step_fn()
        gexec = cap.graph.instantiate()
        return gexec

    # ------ 폴백 ------
    def _one_step():
        cur = _run_fwd(model, plan, X_buf, stream.ptr)
        loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)
        # ✅ 손실값을 고정 버퍼로 복사(있으면)
        if loss_out is not None:
            loss_out[...] = loss_dev
        g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
        if dY is not None:
            dY[...] = dY_tmp
        _zero_bwd_buffers(plan)
        _run_bwd(model, plan, g_in, stream.ptr)
        optimizer_step_fn()

    return GraphExecLike(_one_step, stream)


class TrainGraph:
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
        xb, yb = self._io["X"], self._io["y"]
        with self._stream:  # ✅ 그래프와 동일 스트림에서 H2D/D2D 수행
            xb[...] = cp.asarray(X_dev, dtype=xb.dtype)
            yb[...] = cp.asarray(y_dev, dtype=yb.dtype)

    def launch(self):
        # GraphExecLike와 동일 인터페이스 유지
        self._gexec.launch(self._stream.ptr)
