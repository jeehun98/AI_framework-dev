# python/graph_executor_v2/graph/capture_plan.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cupy as cp

# Sequential 계열 모델의 그래프 캡처용 사전 할당 계획(CapturePlan)을 만든다.
# 레이어별 순전파 출력/프리액티베이션 버퍼(y,z), 역전파 버퍼(gA,gW,gB), 백엔드 워크스페이스(work) 를 준비.


# 내부 GEMM 워크스페이스 유틸 (지연 임포트)
def _ensure_gemm_workspaces(m: int, n: int, *, lt_bytes: int):
    from graph_executor_v2.ops import gemm as gops  # type: ignore
    return gops.ensure_workspaces(m, n, lt_bytes=lt_bytes)

@dataclass
class PerLayerBufs:
    name: str
    y: cp.ndarray                  # forward output
    z: Optional[cp.ndarray]        # pre-activation(optional)
    gA: cp.ndarray                 # grad w.r.t. input
    gW: Optional[cp.ndarray]       # grad w.r.t. weight
    gB: Optional[cp.ndarray]       # grad w.r.t. bias
    work: Any                      # backend-specific (gemm ws or None)

@dataclass
class LossBufs:
    dY: Optional[cp.ndarray]
    out_shape: Tuple[int, ...]

@dataclass
class CapturePlan:
    input_shape: Tuple[int, ...]
    per_layer: List[PerLayerBufs]
    loss: LossBufs

def make_plan_for_sequential(
    model,
    input_shape: Tuple[int, ...],
    *,
    loss_kind: str = "softmax_ce",
    lt_bytes: int = (8 << 20),
) -> CapturePlan:
    """
    Sequential-like 모델을 대상으로 캡처 실행에 필요한 디바이스 버퍼와
    (필요시) GEMM 워크스페이스를 사전할당한다.
    """
    cur = tuple(map(int, input_shape))
    per_layer: List[PerLayerBufs] = []

    for idx, lyr in enumerate(model.layers):
        lname = f"L{idx}:{lyr.__class__.__name__}"
        # 출력 shape 미리 추론
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

        # forward 버퍼
        y = cp.empty(out_shp, dtype=cp.float32)
        need_z = getattr(lyr, "activation", "none") != "none"
        z = cp.empty(out_shp, dtype=cp.float32) if need_z else None

        # backward 버퍼 (Dense 휴리스틱)
        if hasattr(lyr, "W"):
            W = getattr(lyr, "W")
            units = int(W.shape[1])
            gA = cp.empty(cur, dtype=cp.float32)
            gW = cp.empty_like(W)
            gB = cp.empty((1, units), dtype=cp.float32) if hasattr(lyr, "b") else None
            ws = _ensure_gemm_workspaces(cur[0], units, lt_bytes=lt_bytes)
        else:
            # 파라미터 없는 레이어
            gA = cp.empty(cur, dtype=cp.float32)
            gW = None
            gB = None
            ws = None

        per_layer.append(PerLayerBufs(
            name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws
        ))
        cur = out_shp

    if loss_kind == "softmax_ce":
        dY = cp.empty(cur, dtype=cp.float32)
    else:
        dY = None

    return CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(cur))
    )
