# File: python/graph_executor_v2/graph/capture_plan.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Type, Sequence
import os
import cupy as cp

__all__ = [
    "PerLayerBufs", "LossBufs", "CapturePlan",
    "register_planner", "make_plan_for_sequential", "make_plan_for_path",
    "advance_dropout",
]

# ============================================================================
# 이 파일의 역할
# ----------------------------------------------------------------------------
# - CUDA Graph 캡처가 안전하도록, 캡처 구간 이전에 레이어별 버퍼/워크스페이스를
#   사전할당(Pre-allocation)하여 동적 메모리 할당을 제거한다.
# - 정적 경로:  make_plan_for_sequential(model, in_shape)
# - 동적 경로:  make_plan_for_path(path_layers, in_shape)
#
# - 레지스트리 기반 플래너(Planner):
#     레이어 타입 → (PerLayerBufs, out_shape) 생성 함수 등록
#
# - 주요 개선점:
#   * dtype 정책(amp 대응): act/grad dtype을 분리 설정 가능
#   * zero-init 토글: GEV2_PLAN_ZERO_INIT=1 시 버퍼를 zeros로 초기화(디버그 친화)
#   * Conv/BN 레이아웃 대응: channels_last(NHWC)/channels_first(NCHW) 모두 지원
#   * loss dY 동적 질의: loss 객체가 인터페이스 제공 시 shape/dtype 질의
#   * summary(): 디버그용 요약 출력
#   * 에러 메시지/가드 강화
# ============================================================================


# ============================================================
# Core dataclasses
# ============================================================

@dataclass
class PerLayerBufs:
    """단일 레이어의 forward/backward/capture-safe 실행에 필요한 버퍼 묶음.

    필드 설명:
      - name:   디버그용 표시 이름(L{idx}:{LayerClass})
      - y:      forward 출력 버퍼(필수). 다음 레이어 입력으로 이어진다.
      - z:      pre-activation(옵션). act!='none'일 때만 별도 버퍼로 운용.
      - gA:     입력에 대한 gradient (필수). 이전 레이어 backward 입력이 됨.
      - gW:     가중치 gradient (옵션). 파라미터 없는 레이어는 None.
      - gB:     편향 gradient (옵션).
      - work:   backend-specific workspaces (예: GEMM/Conv 임시행렬/인덱스 등)
      - gWx/gWh/dh0: RNN 계열 확장용(역호환 위해 기본 None 유지).
    """
    name: str
    # forward
    y: cp.ndarray
    z: Optional[cp.ndarray]
    # backward
    gA: cp.ndarray
    gW: Optional[cp.ndarray]
    gB: Optional[cp.ndarray]
    # backend-specific workspaces (e.g., GEMM/Conv/Pool indices)
    work: Any
    # ---- RNN extensions (optional; keep defaults for backward-compat) ----
    gWx: Optional[cp.ndarray] = None
    gWh: Optional[cp.ndarray] = None
    dh0: Optional[cp.ndarray] = None

@dataclass
class LossBufs:
    """손실 계산에 필요한 출력 형상 및 dY 버퍼 컨테이너.

    - dY:      손실 backward 입력(softmax_ce 등에서 로짓 도함수 저장용)
               필요 없으면 None
    - out_shape: 로짓(logits) shape
    """
    dY: Optional[cp.ndarray]
    out_shape: Tuple[int, ...]           # shape of logits (for loss)

@dataclass
class CapturePlan:
    """그래프 캡처를 위한 전역 계획(Plan).

    - input_shape: 고정 입력 형상(그래프 캡처의 shape invariant)
    - per_layer:   레이어별 PerLayerBufs 시퀀스
    - loss:        LossBufs (logits shape, dY 버퍼 등)
    - meta:        디버그/추적용 메타데이터(선택)
    """
    input_shape: Tuple[int, ...]
    per_layer: List[PerLayerBufs]
    loss: LossBufs
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Plan의 핵심 정보 요약(디버그/로그용)."""
        try:
            tensors = []
            for p in self.per_layer:
                tensors.append({
                    "name": p.name,
                    "y": tuple(p.y.shape),
                    "z": (tuple(p.z.shape) if isinstance(p.z, cp.ndarray) else None),
                    "gA": tuple(p.gA.shape),
                    "gW": (tuple(p.gW.shape) if isinstance(p.gW, cp.ndarray) else None),
                    "gB": (tuple(p.gB.shape) if isinstance(p.gB, cp.ndarray) else None),
                })
            return {
                "input_shape": tuple(self.input_shape),
                "num_layers": len(self.per_layer),
                "loss_out": tuple(self.loss.out_shape),
                "has_dY": self.loss.dY is not None,
                "tensors": tensors[:8],  # 미리보기 제한
                "meta": dict(self.meta),
            }
        except Exception:
            # 요약 실패해도 테스트는 계속 진행 가능하게
            return {
                "input_shape": tuple(self.input_shape),
                "num_layers": len(self.per_layer),
                "loss_out": tuple(self.loss.out_shape),
                "has_dY": self.loss.dY is not None,
                "meta": dict(self.meta),
            }


# ============================================================
# Low-level helpers
# ============================================================

def _dtype_from_policy(kind: str, policy: Optional[dict]) -> cp.dtype:
    """dtype 정책에서 kind('act'|'grad'|'ws' 등)에 해당하는 dtype을 얻는다."""
    if not policy:
        return cp.float32
    mp = {
        "fp16": cp.float16,
        "bf16": cp.dtype("bfloat16"),
        "fp32": cp.float32,
    }
    return mp.get(str(policy.get(kind, "fp32")).lower(), cp.float32)

def _alloc(shape, *, dtype, zero_init: bool) -> cp.ndarray:
    """디버그/재현성을 위해 zero-init을 토글할 수 있는 할당 헬퍼."""
    return cp.zeros(shape, dtype=dtype) if zero_init else cp.empty(shape, dtype=dtype)

def _ensure_gemm_workspaces(m: int, n: int, *, lt_bytes: int):
    """GEMM(특히 Lt/WMMA 등) 경로의 워크스페이스 확보."""
    from graph_executor_v2.ops import gemm as gops  # type: ignore
    return gops.ensure_workspaces(m, n, lt_bytes=lt_bytes)

def _conv2d_out_hw(
    H: int, W: int, KH: int, KW: int,
    stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int]
) -> Tuple[int, int]:
    """Conv 출력 크기 계산(정수식, PyTorch 동일 규칙)."""
    sH, sW = map(int, stride); pH, pW = map(int, padding); dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
    return H_out, W_out

# zero-init 디버그 토글 (기본 0=off)
_ZERO_INIT = bool(int(os.getenv("GEV2_PLAN_ZERO_INIT", "0")))

# 외부에서 dtype 정책을 전달하지 않으면 None → 모두 fp32
# 예: {"act":"fp16", "grad":"fp32"}
_DEFAULT_DTYPE_POLICY = None


# ============================================================
# Planner registry
# ============================================================

_Planner = Callable[
    [Any, Tuple[int, ...], int, int, Optional[dict], bool],  # (layer, cur_in_shape, layer_index, lt_bytes, dtype_policy, zero_init)
    Tuple[PerLayerBufs, Tuple[int, ...]]  # (per_layer_bufs, out_shape)
]
_PLANNERS: Dict[Type[Any], _Planner] = {}

def register_planner(layer_type: Type[Any]):
    """데코레이터: 레이어 타입에 대한 planner 함수를 레지스트리에 등록."""
    def _wrap(func: _Planner):
        if layer_type in _PLANNERS:
            # 중복 등록 경고(개발 중 중복 import/리로드 대응)
            try:
                import warnings
                warnings.warn(f"[capture_plan] planner for {layer_type.__name__} is being overridden.", RuntimeWarning)
            except Exception:
                pass
        _PLANNERS[layer_type] = func
        return func
    return _wrap

def _find_planner(lyr: Any) -> Optional[_Planner]:
    """정확 매치 우선, 그 다음 isinstance 기반으로 플래너 탐색."""
    t = type(lyr)
    if t in _PLANNERS:
        return _PLANNERS[t]
    for kls, fn in _PLANNERS.items():
        if isinstance(lyr, kls):
            return fn
    return None


# ============================================================
# Generic fallback (param-less / reshape / activation 등)
# ============================================================

def _generic_paramless_planner(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                               dtype_policy: Optional[dict], zero_init: bool):
    """파라미터/WS가 필요 없는 레이어(reshape/act 등)의 기본 플래너.

    - y:  out_shape에 맞춰 act dtype으로 할당
    - z:  없음(activation 분리는 ActivationLayer 전용 플래너에서 처리)
    - gA: 입력과 동일 shape의 grad dtype
    - gW/gB/work: 없음
    """
    lname = f"L{idx}:{lyr.__class__.__name__}"
    try:
        out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
    except Exception as e:
        raise RuntimeError(f"cannot infer output shape for {lname} with cur={cur}: {e}")

    y = _alloc(out_shp, dtype=_dtype_from_policy("act", dtype_policy), zero_init=zero_init)
    z = None
    gA = _alloc(cur, dtype=_dtype_from_policy("grad", dtype_policy), zero_init=zero_init)
    gW = None
    gB = None
    ws = None
    return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp


# ============================================================
# Concrete planners
#   1) Dense (GEMM)
#   2) Conv2D
#   3) Pool2D
#   4) Activation (cupy-only)
#   5) Pad
#   6) BatchNorm2d
#   7) Embedding
#   8) Dropout
#   9) RNN (Elman)
# ============================================================

# 1) Dense
try:
    from graph_executor_v2.layers.dense_gemm import Dense  # type: ignore
except Exception:
    Dense = None  # type: ignore

if Dense is not None:
    @register_planner(Dense)  # type: ignore[misc]
    def _plan_dense(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                    dtype_policy: Optional[dict], zero_init: bool):
        """Dense(GEMM) 레이어 플래너.

        - y:  (N, units)
        - z:  act!='none'인 경우 별도 pre-activation 버퍼
        - gA: (N, K)  (입력과 동일 shape)
        - gW: W와 동일 shape
        - gB: (1, units) 혹은 None
        - work: GEMM Lt/WMMA 등 내부 WS (ensure_workspaces)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname} with cur={cur}: {e}")

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        need_z = getattr(lyr, "activation", "none") != "none"
        z = (_alloc(out_shp, dtype=act_dtype, zero_init=zero_init) if need_z else None)

        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)
        if hasattr(lyr, "W") and lyr.W is not None:
            W = lyr.W
            units = int(W.shape[1])  # assume W:(K, N)
            gW = _alloc(W.shape, dtype=grad_dtype, zero_init=zero_init)
            gB = _alloc((1, units), dtype=grad_dtype, zero_init=zero_init) if getattr(lyr, "b", None) is not None else None
            ws = _ensure_gemm_workspaces(cur[0], units, lt_bytes=lt_bytes)
        else:
            gW = None
            gB = None
            ws = None

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 2) Conv2D
try:
    from graph_executor_v2.layers.conv2d import Conv2D  # type: ignore
except Exception:
    Conv2D = None  # type: ignore

if Conv2D is not None:
    @register_planner(Conv2D)  # type: ignore[misc]
    def _plan_conv2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                     dtype_policy: Optional[dict], zero_init: bool):
        """Conv2D 플래너. NCHW/NHWC 모두 지원.

        - 입력:  (N, Cin, H, W) 또는 (N, H, W, Cin)
        - 출력:  (N, Cout, Ho, Wo) 또는 (N, Ho, Wo, Cout)
        - z:     act!='none'일 때만 별도 버퍼
        - gW:    레이어의 W shape와 동일(있으면), 없으면 추정 shape
        - gB:    bias 사용 시 (Cout,)
        - work:  forward/backward 공용 Conv2DWorkspaces (모든 중간행렬 포함)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: Conv2D expects 4D input, got {cur}")

        # 레이아웃 판단
        channels_last = bool(getattr(lyr, "channels_last", False))
        layout = getattr(lyr, "layout", None) or ("NHWC" if channels_last else "NCHW")
        layout = str(layout).upper()

        # Cur 파싱 (layout별)
        if layout == "NHWC":
            N, H, W, Cin = map(int, cur)
        else:
            N, Cin, H, W = map(int, cur)

        KH, KW = map(int, getattr(lyr, "kernel_size", (1, 1)))
        stride   = getattr(lyr, "stride", (1, 1))
        padding  = getattr(lyr, "padding", (0, 0))
        dilation = getattr(lyr, "dilation", (1, 1))
        groups   = max(1, int(getattr(lyr, "groups", 1)))
        Cout     = int(getattr(lyr, "out_channels"))

        Ho, Wo = _conv2d_out_hw(H, W, KH, KW, stride, padding, dilation)

        # 출력 shape (layout 고려)
        if layout == "NHWC":
            out_shp = (N, Ho, Wo, Cout)
            ga_shape = (N, H, W, Cin)
        else:
            out_shp = (N, Cout, Ho, Wo)
            ga_shape = (N, Cin, H, W)

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "none")
        need_z = (str(act_name).lower() != "none")
        z = (_alloc(out_shp, dtype=act_dtype, zero_init=zero_init) if need_z else None)

        gA = _alloc(ga_shape, dtype=grad_dtype, zero_init=zero_init)

        Wp = getattr(lyr, "W", None)
        if Wp is not None:
            gW = _alloc(Wp.shape, dtype=grad_dtype, zero_init=zero_init)
        else:
            # weight shape 추정: (Cout, Cin/groups, KH, KW)
            gW = _alloc((Cout, Cin // groups, KH, KW), dtype=grad_dtype, zero_init=zero_init)

        with_bias = bool(getattr(lyr, "use_bias", False)) or \
                    (getattr(lyr, "B", None) is not None) or \
                    (getattr(lyr, "bias", None) is not None) or \
                    (getattr(lyr, "b", None) is not None)
        gB = _alloc((Cout,), dtype=grad_dtype, zero_init=zero_init) if with_bias else None

        from graph_executor_v2.ops.conv2d import Conv2DWorkspaces  # type: ignore
        ws = Conv2DWorkspaces()
        # 아래 WS들은 구현체에 맞춰 shape를 설정(기존 코드 유지)
        HWo = Ho * Wo
        K   = (Cin // groups) * KH * KW
        # forward
        ws.dCol   = _alloc((HWo, K),     dtype=grad_dtype, zero_init=zero_init)
        ws.W_KC   = _alloc((K,   Cout),  dtype=grad_dtype, zero_init=zero_init)
        ws.Y_tmp  = _alloc((HWo, Cout),  dtype=grad_dtype, zero_init=zero_init)
        ws.Z_rows = _alloc((HWo, Cout),  dtype=grad_dtype, zero_init=zero_init) if need_z else None
        # backward
        ws.dCol_b  = _alloc((HWo, K), dtype=grad_dtype, zero_init=zero_init)
        ws.dTmp    = _alloc((max(Cout*K, HWo*K),), dtype=grad_dtype, zero_init=zero_init)
        ws.gy_rows = _alloc((Cout, HWo), dtype=grad_dtype, zero_init=zero_init)
        ws.Z_rows_b= _alloc((Cout, HWo), dtype=grad_dtype, zero_init=zero_init)
        ws.W_CK   = _alloc((Cout, K), dtype=grad_dtype, zero_init=zero_init)
        ws.dY_HT  = _alloc((HWo,  Cout), dtype=grad_dtype, zero_init=zero_init)
        ws.dWpack = _alloc((Cout, K), dtype=grad_dtype, zero_init=zero_init)

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 3) Pool2D
try:
    from graph_executor_v2.layers.pool2d import Pool2D as _Pool2DLayer  # type: ignore
except Exception:
    _Pool2DLayer = None  # type: ignore

if _Pool2DLayer is not None:
    @register_planner(_Pool2DLayer)  # type: ignore[misc]
    def _plan_pool2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                     dtype_policy: Optional[dict], zero_init: bool):
        """Pool2D 플래너.

        - max pool인 경우, backward를 위해 인덱스 버퍼(work.indices) 필요
        - y:  (N, C, Ho, Wo) / gA: (N, C, H, W) / gW/gB: 없음
        - channels_last 지원이 필요하면 Pool2D 레이어에 layout 속성을 부여하고 여기서 반영
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: Pool2D expects 4D input, got {cur}")

        channels_last = bool(getattr(lyr, "channels_last", False))
        layout = getattr(lyr, "layout", None) or ("NHWC" if channels_last else "NCHW")
        layout = str(layout).upper()

        if layout == "NHWC":
            N, H, W, C = map(int, cur)
        else:
            N, C, H, W = map(int, cur)

        kH, kW = map(int, getattr(lyr, "kernel_size", (2, 2)))
        stride    = getattr(lyr, "stride", (2, 2))
        padding   = getattr(lyr, "padding", (0, 0))
        dilation  = getattr(lyr, "dilation", (1, 1))
        ceil_mode = bool(getattr(lyr, "ceil_mode", False))
        mode      = str(getattr(lyr, "mode", "max")).lower()

        def _out_hw(H: int, W: int, kH: int, kW: int,
                    s: Tuple[int,int], p: Tuple[int,int], d: Tuple[int,int], ceil: bool):
            sH, sW = s; pH, pW = p; dH, dW = d
            effKH = (kH - 1) * dH + 1
            effKW = (kW - 1) * dW + 1
            aH = H + 2 * pH - effKH
            aW = W + 2 * pW - effKW
            if ceil:
                Ho = (aH >= 0) and ((aH + sH - 1)//sH + 1) or 0
                Wo = (aW >= 0) and ((aW + sW - 1)//sW + 1) or 0
            else:
                Ho = (aH >= 0) and (aH//sH + 1) or 0
                Wo = (aW >= 0) and (aW//sW + 1) or 0
            return max(0, int(Ho)), max(0, int(Wo))

        Ho, Wo = _out_hw(H, W, kH, kW, stride, padding, dilation, ceil_mode)

        if layout == "NHWC":
            out_shp = (N, Ho, Wo, C)
            ga_shape = (N, H, W, C)
        else:
            out_shp = (N, C, Ho, Wo)
            ga_shape = (N, C, H, W)

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z = None
        gA = _alloc(ga_shape, dtype=grad_dtype, zero_init=zero_init)
        gW = None
        gB = None

        work = None
        if mode == "max":
            from graph_executor_v2.layers.pool2d import _Pool2DWork  # local helper
            work = _Pool2DWork()
            # 인덱스는 int32 고정
            if layout == "NHWC":
                work.indices = cp.empty((N, Ho, Wo, C), dtype=cp.int32)
            else:
                work.indices = cp.empty((N, C, Ho, Wo), dtype=cp.int32)

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=work), out_shp

# 4) Activation (cupy only, shape-preserving)
try:
    from graph_executor_v2.layers.activations import ActivationLayer  # type: ignore
except Exception:
    ActivationLayer = None  # type: ignore

if ActivationLayer is not None:
    @register_planner(ActivationLayer)  # type: ignore[misc]
    def _plan_activation(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                         dtype_policy: Optional[dict], zero_init: bool):
        """Shape-preserving activation(별도 파라미터 없음)."""
        lname = f"L{idx}:{lyr.__class__.__name__}"
        out_shp = tuple(map(int, cur))
        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y  = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z  = None
        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)
        gW = None
        gB = None
        ws = None
        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 5) Pad (constant pad; param-less, but shape-change)
try:
    from graph_executor_v2.layers.pad import Pad as _PadLayer  # type: ignore
except Exception:
    _PadLayer = None  # type: ignore

if _PadLayer is not None:
    @register_planner(_PadLayer)  # type: ignore[misc]
    def _plan_pad(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                  dtype_policy: Optional[dict], zero_init: bool):
        """Pad 레이어(파라미터 없음, shape 변경)."""
        lname = f"L{idx}:{lyr.__class__.__name__}"
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname} with cur={cur}: {e}")

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z = None
        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)
        gW = None
        gB = None
        ws = None
        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 6) BatchNorm2d
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d  # type: ignore
except Exception:
    _BN2d = None  # type: ignore

if _BN2d is not None:
    @register_planner(_BN2d)  # type: ignore[misc]
    def _plan_bn2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                   dtype_policy: Optional[dict], zero_init: bool):
        """BatchNorm2d 플래너.

        - forward: y[cur], z=None
        - backward: gA[cur], gW[gamma]/gB[beta] (affine=True일 때)
        - X_saved(prev_y)는 graph_exec에서 prev_y로 전달(BN 제약)
        - channels_last 대응
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: BN2d expects 4D input, got {cur}")

        channels_last = bool(getattr(lyr, "channels_last", False))
        layout = getattr(lyr, "layout", None) or ("NHWC" if channels_last else "NCHW")
        layout = str(layout).upper()
        out_shp = tuple(map(int, cur))

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z = None
        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)

        if getattr(lyr, "affine", True):
            if layout == "NHWC":
                C = int(cur[3])
            else:
                C = int(cur[1])
            gW = _alloc((C,), dtype=grad_dtype, zero_init=zero_init)  # dgamma
            gB = _alloc((C,), dtype=grad_dtype, zero_init=zero_init)  # dbeta
        else:
            gW = None
            gB = None
        ws = None
        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 7) Embedding
try:
    from graph_executor_v2.layers.embedding import Embedding  # type: ignore
except Exception:
    Embedding = None  # type: ignore

if Embedding is not None:
    @register_planner(Embedding)  # type: ignore[misc]
    def _plan_embedding(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                        dtype_policy: Optional[dict], zero_init: bool):
        """Embedding 플래너.

        - 입력: int32 indices (N,L) 또는 (L,)
        - 출력: float32/정책에 따른 act dtype (N,L,D) 또는 (L,D)
        - backward: gA는 보통 필요 없으나, 엔진이 carrier를 기대할 수 있어 유지(필요 없다면 None로 축소 가능)
        - gW: (V,D), gB: 없음
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"

        if len(cur) not in (1, 2):
            raise RuntimeError(f"{lname}: Embedding expects 1D/2D indices, got {cur}")

        V = int(getattr(lyr, "num_embeddings"))
        D = int(getattr(lyr, "embedding_dim"))

        if len(cur) == 2:
            N, L = map(int, cur)
            out_shp = (N, L, D)
        else:
            L = int(cur[0])
            out_shp = (L, D)

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z = None
        # NOTE: 런타임이 dX를 참조하지 않는다면 gA=None로 축소 가능
        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)
        gW = _alloc((V, D), dtype=grad_dtype, zero_init=zero_init)
        gB = None
        ws = None

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), tuple(out_shp)

# 8) Dropout (mask만 마련하는 소형 work)
try:
    from graph_executor_v2.layers.dropout import Dropout as _DropLayer  # type: ignore
except Exception:
    _DropLayer = None  # type: ignore

if _DropLayer is not None:
    class _DropoutWork:
        """Dropout용 내부 work: mask 버퍼 + rng_state(카운터/시드/오프셋 등 포함)."""
        __slots__ = ("mask", "rng_state")
        def __init__(self, shape):
            self.mask = cp.empty(shape, dtype=cp.int32)  # fwd에서 채워지고 bwd에서 재사용
            # rng_state는 커널이 해석할 수 있는 구조체/튜플을 가정
            self.rng_state = {"counter_base": 0, "seed": 0, "offset": 0}

    @register_planner(_DropLayer)  # type: ignore[misc]
    def _plan_dropout(lyr, cur, idx, lt_bytes, dtype_policy, zero_init):
        lname = f"L{idx}:{lyr.__class__.__name__}"
        out_shp = tuple(map(int, cur))  # shape 보존형 레이어

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y  = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)
        z  = None
        gA = _alloc(cur, dtype=grad_dtype, zero_init=zero_init)
        gW = None
        gB = None

        ws = _DropoutWork(out_shp)
        # 레이어가 seed 속성을 제공하면 초기 반영
        seed = getattr(lyr, "seed", None)
        if seed is not None:
            ws.rng_state["seed"] = int(seed)
        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 9) RNN (Elman)
try:
    from graph_executor_v2.layers.rnn import RNN  # type: ignore
except Exception:
    RNN = None  # type: ignore

if RNN is not None:
    @register_planner(RNN)  # type: ignore[misc]
    def _plan_rnn(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int,
                  dtype_policy: Optional[dict], zero_init: bool):
        """RNN(Elman) 플래너.

        - 입력: X (N,T,I)
        - 출력: y (N,T,H), z(optional) = pre-activation
        - backward: dX(N,T,I), gWx(I,H), gWh(H,H), gB(H,), dh0(N,H)
        - work: RnnWorkspaces (fwd/bwd 공용 버퍼)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"

        if len(cur) != 3:
            raise RuntimeError(f"{lname}: RNN expects 3D input (N,T,I), got {cur}")
        N, T, I = map(int, cur)
        H = int(getattr(lyr, "hidden_size"))

        out_shp = (N, T, H)

        act_dtype = _dtype_from_policy("act", dtype_policy)
        grad_dtype = _dtype_from_policy("grad", dtype_policy)

        y = _alloc(out_shp, dtype=act_dtype, zero_init=zero_init)

        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "tanh")
        need_z = (str(act_name).lower() != "none")
        z = (_alloc(out_shp, dtype=act_dtype, zero_init=zero_init) if need_z else None)

        gA  = _alloc((N, T, I), dtype=grad_dtype, zero_init=zero_init)  # dX
        gW  = None
        gWx = _alloc((I, H),     dtype=grad_dtype, zero_init=zero_init)
        gWh = _alloc((H, H),     dtype=grad_dtype, zero_init=zero_init)

        with_bias = bool(getattr(lyr, "with_bias", False)) or \
                    (getattr(lyr, "b", None) is not None) or \
                    (getattr(lyr, "bias", None) is not None)
        gB  = _alloc((H,), dtype=grad_dtype, zero_init=zero_init) if with_bias else None
        dh0 = _alloc((N, H), dtype=grad_dtype, zero_init=zero_init)

        from graph_executor_v2.ops.rnn import RnnWorkspaces  # type: ignore
        ws = RnnWorkspaces()
        # forward workspaces
        ws.XH_cat   = _alloc((N, I+H), dtype=grad_dtype, zero_init=zero_init)
        ws.Y_rows   = _alloc((N, H),   dtype=grad_dtype, zero_init=zero_init)
        ws.W_cat    = _alloc((I+H, H), dtype=grad_dtype, zero_init=zero_init)
        ws.Z_rows_f = _alloc((N, H),   dtype=grad_dtype, zero_init=zero_init) if need_z else None
        # backward workspaces
        ws.XH_cat_b = _alloc((N, I+H), dtype=grad_dtype, zero_init=zero_init)
        ws.G_rows   = _alloc((N, H),   dtype=grad_dtype, zero_init=zero_init)
        ws.Z_rows_b = _alloc((N, H),   dtype=grad_dtype, zero_init=zero_init)
        ws.W_cat_b  = _alloc((I+H, H), dtype=grad_dtype, zero_init=zero_init)
        ws.dXH_cat  = _alloc((N, I+H), dtype=grad_dtype, zero_init=zero_init)
        ws.dWcat    = _alloc((I+H, H), dtype=grad_dtype, zero_init=zero_init)
        ws.TmpW     = _alloc((I+H, H), dtype=grad_dtype, zero_init=zero_init)

        return (
            PerLayerBufs(
                name=lname, y=y, z=z,
                gA=gA, gW=gW, gB=gB, work=ws,
                gWx=gWx, gWh=gWh, dh0=dh0,
            ),
            out_shp
        )


# ============================================================
# Internal utility used by public planners
# ============================================================

def _plan_over_layers(
    layers: Sequence[Any],
    input_shape: Tuple[int, ...],
    *,
    lt_bytes: int = (8 << 20),
    dtype_policy: Optional[dict] = _DEFAULT_DTYPE_POLICY,
    zero_init: bool = _ZERO_INIT,
) -> Tuple[List[PerLayerBufs], Tuple[int, ...]]:
    """선형 경로 레이어 시퀀스에 대해 순차 플래닝을 수행.

    규칙:
      - 레이어가 build(cur)를 가진 경우 호출(실패는 무시 가능: shape-free build)
      - 레지스트리에 등록된 플래너 우선, 없으면 generic planner
      - 각 단계의 out_shape가 다음 단계의 cur로 연결
    반환:
      - per_layer: PerLayerBufs 리스트
      - 최종 out_shape
    """
    cur = tuple(map(int, input_shape))
    per_layer: List[PerLayerBufs] = []

    for idx, lyr in enumerate(layers):
        # 개별 build 지원(일부 레이어는 shape-independent → 실패 무시)
        if not getattr(lyr, "built", False) and hasattr(lyr, "build"):
            try:
                lyr.build(cur)
            except Exception:
                pass

        planner = _find_planner(lyr)
        if planner is None:
            per, out_shp = _generic_paramless_planner(lyr, cur, idx, lt_bytes, dtype_policy, zero_init)
        else:
            per, out_shp = planner(lyr, cur, idx, lt_bytes, dtype_policy, zero_init)
        per_layer.append(per)
        cur = tuple(map(int, out_shp))

    return per_layer, cur


# ============================================================
# Public API
# ============================================================

def make_plan_for_sequential(
    model,
    input_shape: Tuple[int, ...],
    *,
    loss_kind: str = "softmax_ce",
    loss_obj: Optional[Any] = None,           # ← loss 객체가 dY 요구/shape/dtype을 노출할 수 있음
    lt_bytes: int = (8 << 20),
    dtype_policy: Optional[dict] = _DEFAULT_DTYPE_POLICY,
    zero_init: Optional[bool] = None,         # None이면 환경변수 기반 디폴트(_ZERO_INIT)
) -> CapturePlan:
    """정적 경로용: Sequential 전체 모델에 대한 CapturePlan 생성.

    흐름:
      1) (필요 시) model.build(in_shape)
      2) _plan_over_layers(model.layers, in_shape)
      3) loss_obj 또는 loss_kind에 따라 dY 버퍼 준비
      4) CapturePlan 반환

    참고:
      - 이 단계는 CUDA Graph 캡처가 아니라 '버퍼/WS 사전할당'만 담당.
      - dtype_policy 예: {"act":"fp16", "grad":"fp32"} (미지정 시 모두 fp32)
      - zero_init: 디버그/재현성 위해 zeros로 초기화(메모리/성능 트레이드오프)
    """
    cur = tuple(map(int, input_shape))

    # 필요 시 자동 build
    if not getattr(model, "built", False):
        if hasattr(model, "build"):
            model.build(cur)

    layers = list(getattr(model, "layers", []))
    per_layer, out_shape = _plan_over_layers(
        layers, cur, lt_bytes=lt_bytes,
        dtype_policy=dtype_policy, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init)
    )

    # loss dY: loss_obj가 인터페이스 제공 시 우선 사용
    dY = None
    if loss_obj is not None and hasattr(loss_obj, "needs_dy") and callable(loss_obj.needs_dy) and loss_obj.needs_dy():
        # dy_shape(out_shape) / dy_dtype 속성/메서드 관용 지원
        if hasattr(loss_obj, "dy_shape") and callable(loss_obj.dy_shape):
            dy_shape = tuple(map(int, loss_obj.dy_shape(out_shape)))
        else:
            dy_shape = tuple(out_shape)
        dy_dtype = getattr(loss_obj, "dy_dtype", cp.float32)
        dY = _alloc(dy_shape, dtype=dy_dtype, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init))
    elif loss_kind == "softmax_ce":
        dY = _alloc(out_shape, dtype=cp.float32, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init))

    plan = CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(out_shape)),
        meta={"dtype_policy": dtype_policy or {"act": "fp32", "grad": "fp32"}}
    )
    return plan


def make_plan_for_path(
    path_layers: Sequence[Any],
    input_shape: Tuple[int, ...],
    *,
    loss_kind: str = "softmax_ce",
    loss_obj: Optional[Any] = None,
    lt_bytes: int = (8 << 20),
    dtype_policy: Optional[dict] = _DEFAULT_DTYPE_POLICY,
    zero_init: Optional[bool] = None,
) -> CapturePlan:
    """동적 경로용: 평탄화된 '선형 경로' 레이어 리스트에 대한 CapturePlan 생성.

    - Sequential 컨테이너를 거치지 않고도 직접 사용 가능(then/else 블록 등)
    - 내부 동작은 make_plan_for_sequential과 동일하되, 대상 레이어 시퀀스만 다름
    """
    cur = tuple(map(int, input_shape))
    per_layer, out_shape = _plan_over_layers(
        path_layers, cur, lt_bytes=lt_bytes,
        dtype_policy=dtype_policy, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init)
    )

    dY = None
    if loss_obj is not None and hasattr(loss_obj, "needs_dy") and callable(loss_obj.needs_dy) and loss_obj.needs_dy():
        if hasattr(loss_obj, "dy_shape") and callable(loss_obj.dy_shape):
            dy_shape = tuple(map(int, loss_obj.dy_shape(out_shape)))
        else:
            dy_shape = tuple(out_shape)
        dy_dtype = getattr(loss_obj, "dy_dtype", cp.float32)
        dY = _alloc(dy_shape, dtype=dy_dtype, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init))
    elif loss_kind == "softmax_ce":
        dY = _alloc(out_shape, dtype=cp.float32, zero_init=_ZERO_INIT if zero_init is None else bool(zero_init))

    plan = CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(out_shape)),
        meta={"dtype_policy": dtype_policy or {"act": "fp32", "grad": "fp32"}}
    )
    return plan


# ============================================================
# Utilities
# ============================================================

def advance_dropout(plan: CapturePlan, seed_bump: int = 1) -> None:
    """Replay 루프에서 Dropout 마스크 재생성 유도.

    - 각 레이어의 work.rng_state.counter_base/offset 등을 증가시켜
      PRNG offset에 반영(커널 구현에 맞춰 해석)
    - 정책적으로 꺼두고 싶으면 seed_bump=0 사용.
    """
    if seed_bump == 0:
        return
    for per in plan.per_layer:
        work = getattr(per, "work", None)
        if work is None:
            continue
        # 최신 Dropout work 기준(rng_state dict)
        rng = getattr(work, "rng_state", None)
        if isinstance(rng, dict):
            try:
                rng["counter_base"] = int(rng.get("counter_base", 0)) + int(seed_bump)
                # 필요 시 offset도 전진
                rng["offset"] = int(rng.get("offset", 0)) + int(seed_bump)
            except Exception:
                pass
            continue
        # 구버전(counter_base만 존재)
        if hasattr(work, "counter_base"):
            try:
                work.counter_base += int(seed_bump)
            except Exception:
                pass
