from __future__ import annotations
from typing import Sequence, Optional, Any
import cupy as cp

from .capture_plan import CapturePlan
from .exec_utils import (
    _ensure_conv2d_ws_for_forward,
    _ensure_conv2d_ws_for_backward,
    _zero_bwd_buffers,
    _loss_forward,
    nvtx_range,
)
from graph_executor_v2.layers.conv2d import Conv2D
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d
except Exception:
    _BN2d = None


class GraphRuntime:
    def __init__(self, stream: cp.cuda.Stream):
        self._stream = stream

    def run_step(
        self,
        *,
        layers: Sequence[Any],
        plan: CapturePlan,
        loss_fn,
        optimizer_step_fn,
        X_buf: cp.ndarray,
        y_buf: cp.ndarray,
        loss_out: Optional[cp.ndarray],
        capture: bool,
    ):
        # 모든 커널/연산을 동일 CUDA 스트림에서 실행
        with self._stream:

            # ---------- forward ----------
            cur = X_buf
            for i, lyr in enumerate(layers):
                per = plan.per_layer[i]
                ybuf = per.y
                zbuf = per.z

                ws_local = None
                if isinstance(lyr, Conv2D):
                    ws_local = _ensure_conv2d_ws_for_forward(per, lyr, tuple(map(int, cur.shape)))

                try:
                    lyr.forward_into(
                        cur,
                        out=ybuf,
                        z_out=zbuf,
                        work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                        stream=self._stream.ptr,
                    )
                except TypeError:
                    # 최소 호환 시그니처
                    lyr.forward_into(cur, out=ybuf, stream=self._stream.ptr)

                cur = ybuf

            # ---------- loss forward ----------
            loss_dev, dY_tmp = _loss_forward(loss_fn, cur, y_buf)
            if loss_out is not None:
                loss_out[...] = loss_dev

            dY = plan.loss.dY
            g_in = dY if (dY is not None and dY.shape == dY_tmp.shape) else dY_tmp
            if dY is not None:
                dY[...] = dY_tmp

            # ---------- zero grads ----------
            _zero_bwd_buffers(plan)

            # ---------- backward ----------
            for ridx, lyr in enumerate(reversed(layers)):
                i = len(layers) - 1 - ridx
                per = plan.per_layer[i]

                ws_local = None
                if isinstance(lyr, Conv2D):
                    ws_local = _ensure_conv2d_ws_for_backward(per, lyr)

                # BN2d 특수 경로: X_saved 필요
                is_bn = (_BN2d is not None and isinstance(lyr, _BN2d))
                if is_bn:
                    # 첫 레이어면 이전 출력이 없으므로 입력 버퍼(X_buf)를 사용
                    prev_y = plan.per_layer[i - 1].y if i - 1 >= 0 else X_buf
                    lyr.backward_into(
                        g_in,
                        gA_out=per.gA,
                        gW_out=per.gW,
                        gB_out=per.gB,
                        X_saved=prev_y,
                        stream=self._stream.ptr,
                    )
                    g_in = per.gA
                    continue

                if per.gW is not None:
                    # 파라미터 있는 레이어
                    try:
                        lyr.backward_into(
                            g_in,
                            gA_out=per.gA,
                            gW_out=per.gW,
                            gB_out=per.gB,
                            work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                            stream=self._stream.ptr,
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
                                stream=self._stream.ptr,
                            )
                        except TypeError:
                            lyr.backward_into(
                                g_in,
                                gA_out=per.gA,
                                gW_out=per.gW,
                                gB_out=per.gB,
                                stream=self._stream.ptr,
                            )
                else:
                    # 입력 그래드만 있는 레이어
                    try:
                        lyr.backward_into(
                            g_in,
                            gA_out=per.gA,
                            work=(ws_local if ws_local is not None else getattr(per, "work", None)),
                            stream=self._stream.ptr,
                        )
                    except TypeError:
                        lyr.backward_into(g_in, gA_out=per.gA, stream=self._stream.ptr)

                g_in = per.gA

            # ---------- optimizer ----------
            optimizer_step_fn()
