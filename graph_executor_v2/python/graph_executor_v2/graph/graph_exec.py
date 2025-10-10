# python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional
import cupy as cp
from .capture_plan import CapturePlan

class GraphExecLike:
    def __init__(self, launch_fn, stream: cp.cuda.Stream):
        self._launch = launch_fn
        self._stream = stream
    def launch(self, stream_ptr=None):
        with self._stream:
            self._launch()

def _zero_bwd_buffers(plan: CapturePlan):
    for p in plan.per_layer:
        if p.gA is not None: p.gA.fill(0)
        if p.gW is not None: p.gW.fill(0)
        if p.gB is not None: p.gB.fill(0)

def _run_fwd(model, plan: CapturePlan, X, stream_ptr: Optional[int]):
    """고정 입력 X에서 시작해 레이어별 forward_into 실행."""
    cur = X
    for i, lyr in enumerate(model.layers):
        ybuf = plan.per_layer[i].y
        zbuf = plan.per_layer[i].z
        lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream_ptr)
        cur = ybuf
    return cur

def _run_bwd(model, plan: CapturePlan, g_in, stream_ptr: Optional[int]):
    for ridx, lyr in enumerate(reversed(model.layers)):
        i = len(model.layers) - 1 - ridx
        per = plan.per_layer[i]
        ws = per.work
        if per.gW is not None:
            lyr.backward_into(
                g_in,
                gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                work_dZ=(ws.dZ if ws is not None else None),
                lt_workspace=(ws.lt_ws if (ws is not None and ws.lt_ws is not None) else None),
                stream=stream_ptr
            )
        else:
            lyr.backward_into(  # type: ignore
                g_in, gA_out=per.gA,
                work_dZ=None, lt_workspace=None, stream=stream_ptr
            )
        g_in = per.gA
    return g_in

def record_step_graph(
    model,
    loss_fn,
    optimizer_step_fn,
    plan: CapturePlan,
    *,
    X_buf: cp.ndarray,          # ✅ 고정 입력 버퍼
    y_buf: cp.ndarray,          # ✅ 고정 라벨 버퍼
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
        # FWD
        cur = _run_fwd(model, plan, X_buf, stream.ptr)
        # LOSS (고정 y_buf 사용)
        loss_dev, dY_tmp = loss_fn.forward(cur, y_buf, return_scalar=False)
        g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
        if dY is not None:
            dY[...] = dY_tmp
        # BWD
        _zero_bwd_buffers(plan)
        _run_bwd(model, plan, g_in, stream.ptr)
        # OPT
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
