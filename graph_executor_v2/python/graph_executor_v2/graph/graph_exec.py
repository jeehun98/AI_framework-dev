# python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional
import cupy as cp
from .capture_plan import CapturePlan

# Conv2D / WS 유틸
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.ops import conv2d as convops


class GraphExecLike:
    def __init__(self, launch_fn, stream: cp.cuda.Stream):
        self._launch = launch_fn
        self._stream = stream
    def launch(self, stream_ptr=None):
        with self._stream:
            self._launch()


# ---------------- shape / workspace helpers ----------------
def _out_hw(H: int, W: int, KH: int, KW: int,
            stride: tuple[int, int],
            padding: tuple[int, int],
            dilation: tuple[int, int]) -> tuple[int, int]:
    sH, sW = stride; pH, pW = padding; dH, dW = dilation
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
    return H_out, W_out


def _alloc_conv2d_ws(HWo: int, K: int, Cout: int) -> convops.Conv2DWorkspaces:
    ws = convops.Conv2DWorkspaces()
    # Forward (옵션 A: 항상 Z_rows 필요)
    ws.dCol   = cp.empty((HWo, K),    dtype=cp.float32)
    ws.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
    ws.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
    ws.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)
    # Backward (공통)
    ws.dCol_b  = cp.empty((HWo, K),    dtype=cp.float32)
    ws.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
    ws.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
    ws.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
    # Backward 옵션 (gX, gW)
    ws.W_CK    = cp.empty((Cout, K),   dtype=cp.float32)
    ws.dY_HT   = cp.empty((HWo,  Cout),dtype=cp.float32)
    ws.dWpack  = cp.empty((Cout, K),   dtype=cp.float32)
    return ws


def _ensure_conv2d_ws_for_forward(per, lyr: Conv2D, cur_shape: tuple[int, int, int, int]) -> convops.Conv2DWorkspaces:
    """
    Forward 직전에 Conv2D WS를 준비하여 반환.
    per에 저장이 가능하면 저장도 시도하지만, 저장 실패해도 반환값을 즉시 사용하므로 안전.
    """
    # 이미 있다면 그대로 사용
    ws = getattr(per, "work", None)
    if ws is not None:
        return ws

    N, Cin, H, W = map(int, cur_shape)
    KH, KW = lyr.kernel_size
    Cout   = int(lyr.out_channels)
    groups = int(lyr.groups)

    H_out, W_out = _out_hw(H, W, KH, KW, lyr.stride, lyr.padding, lyr.dilation)
    HWo = H_out * W_out
    K   = (Cin // groups) * KH * KW

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

    # lyr.W: (Cout, Cin, KH, KW)
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
        if p.gA is not None: p.gA.fill(0)
        if p.gW is not None: p.gW.fill(0)
        if p.gB is not None: p.gB.fill(0)


# ---------------- forward / backward runners ----------------
def _run_fwd(model, plan: CapturePlan, X, stream_ptr: Optional[int]):
    """
    고정 입력 X에서 시작해 레이어별 forward_into 실행.
    Conv2D는 WS를 즉시 생성/전달한다(옵션 A 대응).
    """
    cur = X
    for i, lyr in enumerate(model.layers):
        per  = plan.per_layer[i]
        ybuf = per.y
        zbuf = per.z

        # Conv2D인 경우: WS를 즉시 준비하고 지역변수로 전달
        ws_local = None
        if isinstance(lyr, Conv2D):
            ws_local = _ensure_conv2d_ws_for_forward(per, lyr, cur.shape)

        try:
            lyr.forward_into(cur, out=ybuf, z_out=zbuf,
                             work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                             stream=stream_ptr)
        except TypeError:
            # Dense 등: work 인자를 받지 않는 시그니처
            lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream_ptr)

        cur = ybuf
    return cur


def _run_bwd(model, plan: CapturePlan, g_in, stream_ptr: Optional[int]):
    """
    레이어별 backward_into 호출.
    Conv2D의 경우 per.work가 없더라도 per.y/lyr.W로 크기 재계산해 WS 즉시 생성.
    """
    for ridx, lyr in enumerate(reversed(model.layers)):
        i = len(model.layers) - 1 - ridx
        per = plan.per_layer[i]

        ws_local = None
        if isinstance(lyr, Conv2D):
            ws_local = _ensure_conv2d_ws_for_backward(per, lyr)

        if per.gW is not None:
            # 파라미터가 있는 레이어(예: Dense, Conv2D)
            try:
                lyr.backward_into(
                    g_in,
                    gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                    work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                    stream=stream_ptr
                )
            except TypeError:
                try:
                    # Dense 전용 시그니처(레거시 GEMM): work_dZ / lt_workspace
                    ws = (ws_local if ws_local is not None else getattr(per, "work", None))
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                        work_dZ=(ws.dZ if ws is not None and hasattr(ws, "dZ") else None),
                        lt_workspace=(ws.lt_ws if ws is not None and hasattr(ws, "lt_ws") else None),
                        stream=stream_ptr
                    )
                except TypeError:
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                        stream=stream_ptr
                    )
        else:
            # 파라미터 없는 레이어(활성화, reshape 등)
            try:
                lyr.backward_into(g_in, gA_out=per.gA,
                                  work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                                  stream=stream_ptr)
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
):
    """
    fwd → loss → bwd → opt 한 스텝을 CUDA Graph로 녹화해 실행자 반환.
    Graph 미지원이면 Pseudo 실행자 반환.
    """
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)

    dY = plan.loss.dY

    # ------ 워밍업 ------
    with stream:
        cur = _run_fwd(model, plan, X_buf, stream.ptr)
        loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)
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
    def logits(self): return self._io["logits"]
    @property
    def X_buf(self):  return self._io["X"]
    @property
    def y_buf(self):  return self._io["y"]
    def set_batch(self, X_dev, y_dev):
        xb, yb = self._io["X"], self._io["y"]
        xb[...] = cp.asarray(X_dev, dtype=xb.dtype)
        yb[...] = cp.asarray(y_dev, dtype=yb.dtype)
    def launch(self):
        self._gexec.launch(self._stream.ptr)
