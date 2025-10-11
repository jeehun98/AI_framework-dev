from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
import cupy as cp

# ============================================================
# CapturePlan core dataclasses
# ============================================================

__all__ = [
    "PerLayerBufs", "LossBufs", "CapturePlan",
    "register_planner", "make_plan_for_sequential",
]

def _generic_paramless_planner(lyr, cur, idx, lt_bytes):
    lname = f"L{idx}:{lyr.__class__.__name__}"
    if not hasattr(lyr, "compute_output_shape"):
        raise RuntimeError(f"{lname}: register a planner or implement compute_output_shape()")

@dataclass
class PerLayerBufs:
    name: str
    # forward
    y: cp.ndarray                        # forward output (required)
    z: Optional[cp.ndarray]              # pre-activation (optional)
    # backward
    gA: cp.ndarray                       # grad w.r.t input (required)
    gW: Optional[cp.ndarray]             # grad w.r.t weight (optional)
    gB: Optional[cp.ndarray]             # grad w.r.t bias (optional)
    # backend-specific workspaces (e.g., GEMM/Conv)
    work: Any

@dataclass
class LossBufs:
    dY: Optional[cp.ndarray]
    out_shape: Tuple[int, ...]           # shape of logits (for loss)

@dataclass
class CapturePlan:
    input_shape: Tuple[int, ...]
    per_layer: List[PerLayerBufs]
    loss: LossBufs


# ============================================================
# Low-level helpers
# ============================================================

def _ensure_gemm_workspaces(m: int, n: int, *, lt_bytes: int):
    """
    Dense/GEMM용 워크스페이스 확보 (지연 임포트)
    """
    from graph_executor_v2.ops import gemm as gops  # type: ignore
    return gops.ensure_workspaces(m, n, lt_bytes=lt_bytes)

def _conv2d_out_hw(
    H: int, W: int, KH: int, KW: int,
    stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int]
) -> Tuple[int, int]:
    sH, sW = map(int, stride); pH, pW = map(int, padding); dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
    return H_out, W_out


# ============================================================
# Layer planner registry
#   - 각 레이어 타입별로 plan 함수를 등록하여, 추가/분리 용이하게 구성
# ============================================================

_Planner = Callable[
    [Any, Tuple[int, ...], int, int],  # (layer, cur_in_shape, layer_index, lt_bytes)
    Tuple[PerLayerBufs, Tuple[int, ...]]  # (per_layer_bufs, out_shape)
]
_PLANNERS: Dict[Type[Any], _Planner] = {}

def register_planner(layer_type: Type[Any]):
    """
    @register_planner(MyLayer)
    def plan_mylayer(lyr, cur, idx, lt_bytes): ...
    """
    def _wrap(func: _Planner):
        _PLANNERS[layer_type] = func
        return func
    return _wrap

def _find_planner(lyr: Any) -> Optional[_Planner]:
    # 정확한 타입 매치 우선
    t = type(lyr)
    if t in _PLANNERS:
        return _PLANNERS[t]
    # 상속 계층 탐색(필요시)
    for kls, fn in _PLANNERS.items():
        if isinstance(lyr, kls):
            return fn
    return None


# ============================================================
# Generic fallback planner (param-less or unknown)
# ============================================================

def _generic_paramless_planner(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
    """
    파라미터 없는 레이어(예: 활성화, reshape 등) 기본 플래너.
    - forward: y[out_shape], z(optional if act!=none)
    - backward: gA[cur], no gW/gB
    - work: None
    """
    lname = f"L{idx}:{lyr.__class__.__name__}"
    try:
        out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
    except Exception as e:
        raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

    y = cp.empty(out_shp, dtype=cp.float32)
    need_z = getattr(lyr, "activation", "none") != "none"
    z = cp.empty(out_shp, dtype=cp.float32) if need_z else None

    gA = cp.empty(cur, dtype=cp.float32)
    gW = None
    gB = None
    ws  = None

    return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp


# ============================================================
# Concrete planners
#   1) Dense (GEMM 기반)
#   2) Conv2D (있으면 활성)
# ============================================================

# 1) Dense/GEMM (layers.dense_gemm.Dense)
try:
    from graph_executor_v2.layers.dense_gemm import Dense  # type: ignore
except Exception:
    Dense = None  # type: ignore

if Dense is not None:
    @register_planner(Dense)  # type: ignore[misc]
    def _plan_dense(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """
        Dense 플래너:
          - forward y[out], z[out if activation!=none]
          - backward gA[cur], gW[W.shape], gB[(1, units)]
          - work = gemm.ensure_workspaces(B, units)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

        y = cp.empty(out_shp, dtype=cp.float32)
        need_z = getattr(lyr, "activation", "none") != "none"
        z = cp.empty(out_shp, dtype=cp.float32) if need_z else None

        # backward buffers
        gA = cp.empty(cur, dtype=cp.float32)
        if hasattr(lyr, "W") and lyr.W is not None:
            W = lyr.W
            units = int(W.shape[1])
            gW = cp.empty_like(W)
            gB = cp.empty((1, units), dtype=cp.float32) if getattr(lyr, "b", None) is not None else None
            ws = _ensure_gemm_workspaces(cur[0], units, lt_bytes=lt_bytes)
        else:
            gW = None
            gB = None
            ws = None

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp


# 2) Conv2D (layers.conv2d.Conv2D + ops.conv2d.Conv2DWorkspaces)
try:
    from graph_executor_v2.layers.conv2d import Conv2D  # type: ignore
except Exception:
    Conv2D = None  # type: ignore

if Conv2D is not None:
    @register_planner(Conv2D)  # type: ignore[misc]
    def _plan_conv2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """
        Conv2D 플래너:
          - forward: y[(N,Cout,H_out,W_out)], z(활성화 시)
          - backward: gA[(N,Cin,H,W)], gW[(Cout,Cin,KH,KW)], gB[(Cout,)](옵션)
          - work: Conv2DWorkspaces (HWo/K/Cout 기반)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: Conv2D expects 4D input (N,C,H,W), got {cur}")
        N, Cin, H, W = map(int, cur)
        KH, KW = map(int, getattr(lyr, "kernel_size", (1, 1)))
        stride = getattr(lyr, "stride", (1, 1))
        padding = getattr(lyr, "padding", (0, 0))
        dilation = getattr(lyr, "dilation", (1, 1))
        groups = max(1, int(getattr(lyr, "groups", 1)))
        Cout = int(getattr(lyr, "out_channels"))

        H_out, W_out = _conv2d_out_hw(H, W, KH, KW, stride, padding, dilation)
        out_shp = (N, Cout, H_out, W_out)
        y = cp.empty(out_shp, dtype=cp.float32)

        need_z = (getattr(lyr, "activation", "none") != "none")
        z = y if not need_z else cp.empty_like(y)

        gA = cp.empty((N, Cin, H, W), dtype=cp.float32)
        gW = cp.empty((Cout, Cin, KH, KW), dtype=cp.float32)

        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "none")
        need_z = (str(act_name).lower() != "none")

        with_bias = (
            bool(getattr(lyr, "with_bias", False)) or
            (getattr(lyr, "B", None) is not None) or
            (getattr(lyr, "bias", None) is not None)
        )
        gB = cp.empty((Cout,), dtype=cp.float32) if with_bias else None

        # workspaces (HWo/K/Cout 기반)
        try:
            from graph_executor_v2.ops.conv2d import Conv2DWorkspaces  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"{lname}: Conv2D ops binding not available; build _ops_conv2d first."
            ) from e
        ws = Conv2DWorkspaces()
        # forward WS
        HWo = H_out * W_out
        K   = (Cin // groups) * KH * KW
        ws.dCol   = cp.empty((HWo, K),     dtype=cp.float32)
        ws.W_KC   = cp.empty((K,   Cout),  dtype=cp.float32)
        ws.Y_tmp  = cp.empty((HWo, Cout),  dtype=cp.float32)
        ws.Z_rows = cp.empty((HWo, Cout),  dtype=cp.float32) if need_z else None
        # backward common
        ws.dCol_b  = cp.empty((HWo, K), dtype=cp.float32)
        ws.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
        ws.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
        ws.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
        # backward opt
        ws.W_CK   = cp.empty((Cout, K), dtype=cp.float32)
        ws.dY_HT  = cp.empty((HWo,  Cout), dtype=cp.float32)
        ws.dWpack = cp.empty((Cout, K), dtype=cp.float32)

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp


# ============================================================
# Public API
# ============================================================

def make_plan_for_sequential(
    model,
    input_shape: Tuple[int, ...],
    *,
    loss_kind: str = "softmax_ce",
    lt_bytes: int = (8 << 20),
) -> CapturePlan:
    """
    Sequential-like 모델의 그래프 캡처용 버퍼/워크스페이스를 사전할당.
    - 레이어별 plan 함수는 registry에 등록되어 있으며,
      미등록 타입은 generic(파라미터 없음) 플래너로 처리.
    """
    cur = tuple(map(int, input_shape))

    # ✅ 필요 시 자동 build
    if not getattr(model, "built", False):
        if hasattr(model, "build"):
            model.build(cur)

    per_layer: List[PerLayerBufs] = []

    for idx, lyr in enumerate(getattr(model, "layers", [])):
        planner = _find_planner(lyr)
        if planner is None:
            # 파라미터 없는 레이어로 가정 (활성화/reshape 등)
            per, out_shp = _generic_paramless_planner(lyr, cur, idx, lt_bytes)
        else:
            per, out_shp = planner(lyr, cur, idx, lt_bytes)
        per_layer.append(per)
        cur = tuple(map(int, out_shp))


    # 끝부분, loss dY 준비
    if loss_kind == "softmax_ce":
        dY = cp.empty(tuple(cur), dtype=cp.float32)  # ✅ FP32로 고정
    else:
        dY = None

    return CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(cur)),
    )
