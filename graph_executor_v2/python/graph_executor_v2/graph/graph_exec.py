# python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional
import cupy as cp
from .capture_plan import CapturePlan

# 한 스텝 학습(fwd→loss→bwd→opt) 을 CUDA Graph에 녹화하고 실행자를 반환.

class GraphExecLike:
    def __init__(self, launch_fn, stream: cp.cuda.Stream):
        self._launch = launch_fn
        self._stream = stream
    def launch(self, stream_ptr=None):
        with self._stream:
            self._launch()

# 누적 방지를 위해 gA/gW/gB를 0으로 초기화.
def _zero_bwd_buffers(plan: CapturePlan):
    for p in plan.per_layer:
        if p.gA is not None: p.gA.fill(0)
        if p.gW is not None: p.gW.fill(0)
        if p.gB is not None: p.gB.fill(0)

# X에서 시작해 각 레이어의 forward_into 호출.
def _run_fwd(model, plan: CapturePlan, X, stream_ptr: Optional[int]):
    """
    고정 입력 X에서 시작해 레이어별 forward_into 실행.
    레이어별로 필요한 workspace 키워드가 다르므로, 우선 work를 넘겨보고
    TypeError면 최소 인자 집합으로 재시도한다.
    """
    cur = X
    for i, lyr in enumerate(model.layers):
        per  = plan.per_layer[i]
        ybuf = per.y
        zbuf = per.z
        # 1) 일반 경로: work 전달(Conv2D 등)
        try:
            lyr.forward_into(cur, out=ybuf, z_out=zbuf, work=getattr(per, "work", None), stream=stream_ptr)
        except TypeError:
            # 2) Dense 등: work 인자를 받지 않는 시그니처
            lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream_ptr)
        cur = ybuf
    return cur

# g_in에서 역순으로 backward_into 호출.
def _run_bwd(model, plan: CapturePlan, g_in, stream_ptr: Optional[int]):
    """
    레이어별로 backward_into 시그니처가 다르다.
    - Dense: backward_into(g_in, gA_out, gW_out, gB_out, work_dZ=..., lt_workspace=..., stream=...)
    - Conv2D: backward_into(X, W, gY, Z, ..., gX_out, gW_out, gB_out, work=..., stream=...) (레이어 구현에 따라 다름)
    본 프레임워크의 레이어 래퍼는 최소한 아래 두 형태 중 하나를 만족하도록 설계한다:
      A) backward_into(g_in, gA_out=..., gW_out=..., gB_out=..., work=..., stream=...)
      B) backward_into(g_in, gA_out=..., gW_out=..., gB_out=..., stream=...)
    """
    for ridx, lyr in enumerate(reversed(model.layers)):
        i = len(model.layers) - 1 - ridx
        per = plan.per_layer[i]
        ws  = getattr(per, "work", None)

        if per.gW is not None:
            # 파라미터가 있는 레이어(예: Dense, Conv2D)
            try:
                # 경로 A: work 키워드 사용
                lyr.backward_into(
                    g_in,
                    gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                    work=ws, stream=stream_ptr
                )
            except TypeError:
                try:
                    # Dense 전용 시그니처(레거시 GEMM): work_dZ / lt_workspace
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                        work_dZ=(ws.dZ if ws is not None and hasattr(ws, "dZ") else None),
                        lt_workspace=(ws.lt_ws if ws is not None and hasattr(ws, "lt_ws") else None),
                        stream=stream_ptr
                    )
                except TypeError:
                    # 경로 B: 최소 인자 세트
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA, gW_out=per.gW, gB_out=per.gB,
                        stream=stream_ptr
                    )
        else:
            # 파라미터 없는 레이어(활성화, reshape 등)
            try:
                lyr.backward_into(g_in, gA_out=per.gA, work=ws, stream=stream_ptr)  # 경로 A
            except TypeError:
                lyr.backward_into(g_in, gA_out=per.gA, stream=stream_ptr)           # 경로 B

        g_in = per.gA
    return g_in

def record_step_graph(
    model,
    loss_fn,
    optimizer_step_fn,
    plan: CapturePlan,
    *,
    X_buf: cp.ndarray,          # 고정 입력 버퍼 (예: (N,C,H,W) 그대로)
    y_buf: cp.ndarray,          # 고정 라벨 버퍼
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
