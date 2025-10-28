# File: python/graph_executor_v2/graph/capture_plan.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Type, Sequence
import cupy as cp

__all__ = [
    "PerLayerBufs", "LossBufs", "CapturePlan",
    "register_planner", "make_plan_for_sequential", "make_plan_for_path",
    "advance_dropout",
]

# ============================================================================
# 이 파일의 역할
# ----------------------------------------------------------------------------
# - Graph Capture 단계에서 필요한 "버퍼(activation/grad/workspace)"를
#   레이어별로 미리 할당(사전 계획)하여, 캡처 구간 안에서 동적 할당을
#   피하고(realloc/malloc 금지) CUDA Graph-safe를 보장한다.
# - 정적 경로:  make_plan_for_sequential(model, in_shape)
# - 동적 경로:  make_plan_for_path(path_layers, in_shape)
#
# - 플래너(Planner) 레지스트리:
#     레이어 타입 → (PerLayerBufs, out_shape) 생성 함수 등록
#   미등록 타입은 파라미터 없는 generic planner로 처리한다.
#
# - 확장 포인트:
#   * (TODO) Execution Planner 연동: 여기서 만든 CapturePlan(per_layer.* 버퍼)을
#           ExecPlanner의 노드/에지 메타로 전송해 스케줄링(스트림/이벤트) 계획 가능
#   * (TODO) Mixed precision: dtype 정책을 한 곳에서 관리하도록 확장 가능
# ============================================================================

# ============================================================
# Core dataclasses
# ============================================================

@dataclass
class PerLayerBufs:
    """단일 레이어의 forward/backward/capture-safe 실행에 필요한 버퍼 묶음.

    필드 설명:
      - y:   forward 출력 버퍼(필수). 다음 레이어 입력으로 이어진다.
      - z:   pre-activation(옵션). act!='none'일 때만 별도 버퍼로 운용한다.
      - gA:  입력에 대한 gradient (필수). 이전 레이어 backward 입력이 됨.
      - gW:  가중치 gradient (옵션). 파라미터 없는 레이어는 None.
      - gB:  편향 gradient (옵션).
      - work: backend-specific workspaces (e.g., GEMM/Conv 임시행렬/인덱스 등)
      - gWx/gWh/dh0: RNN 계열 확장용(역호환 위해 기본 None 유지).
    """
    name: str
    # forward
    y: cp.ndarray                        # forward output (required)
    z: Optional[cp.ndarray]              # pre-activation (optional; act!='none'일 때만 별도 버퍼)
    # backward
    gA: cp.ndarray                       # grad w.r.t input (required)
    gW: Optional[cp.ndarray]             # grad w.r.t weight (optional, single tensor)
    gB: Optional[cp.ndarray]             # grad w.r.t bias (optional)
    # backend-specific workspaces (e.g., GEMM/Conv/Pool indices)
    work: Any
    # ---- RNN extensions (optional; keep defaults for backward-compat) ----
    gWx: Optional[cp.ndarray] = None     # grad w.r.t input weight Wx (I,H)
    gWh: Optional[cp.ndarray] = None     # grad w.r.t recurrent weight Wh (H,H)
    dh0: Optional[cp.ndarray] = None     # grad w.r.t initial hidden state h0 (N,H)

@dataclass
class LossBufs:
    """손실 계산에 필요한 출력 형상 및 dY 버퍼 컨테이너."""
    dY: Optional[cp.ndarray]
    out_shape: Tuple[int, ...]           # shape of logits (for loss)

@dataclass
class CapturePlan:
    """그래프 캡처를 위한 전역 계획(Plan).

    - input_shape: 고정 입력 형상(그래프 캡처의 shape invariant)
    - per_layer:   레이어별 PerLayerBufs 시퀀스
    - loss:        LossBufs (logits shape, dY 버퍼 등)
    """
    input_shape: Tuple[int, ...]
    per_layer: List[PerLayerBufs]
    loss: LossBufs

# ============================================================
# Low-level helpers
# ============================================================

def _ensure_gemm_workspaces(m: int, n: int, *, lt_bytes: int):
    """GEMM(특히 Lt/WMMA 등) 경로의 워크스페이스 확보.

    - 캡처-safe를 위해 필요 용량을 사전에 할당(lt_bytes Hint)
    - 실제 구현은 ops.gemm.ensure_workspaces에 위임
    """
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

# ============================================================
# Planner registry
# ============================================================

_Planner = Callable[
    [Any, Tuple[int, ...], int, int],  # (layer, cur_in_shape, layer_index, lt_bytes)
    Tuple[PerLayerBufs, Tuple[int, ...]]  # (per_layer_bufs, out_shape)
]
_PLANNERS: Dict[Type[Any], _Planner] = {}

def register_planner(layer_type: Type[Any]):
    """데코레이터: 레이어 타입에 대한 planner 함수를 레지스트리에 등록."""
    def _wrap(func: _Planner):
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

def _generic_paramless_planner(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
    """파라미터/WS가 필요 없는 레이어(reshape/act 등)의 기본 플래너.

    - y:  out_shape에 맞춰 float32로 할당
    - z:  없음(activation 분리는 ActivationLayer 전용 플래너에서 처리)
    - gA: 입력과 동일 shape
    - gW/gB/work: 없음
    """
    lname = f"L{idx}:{lyr.__class__.__name__}"
    try:
        out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
    except Exception as e:
        raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

    y = cp.empty(out_shp, dtype=cp.float32)
    z = None
    gA = cp.empty(cur, dtype=cp.float32)
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
    def _plan_dense(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
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
            raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

        y = cp.empty(out_shp, dtype=cp.float32)
        need_z = getattr(lyr, "activation", "none") != "none"
        z = (cp.empty_like(y) if need_z else None)

        gA = cp.empty(cur, dtype=cp.float32)
        if hasattr(lyr, "W") and lyr.W is not None:
            W = lyr.W
            units = int(W.shape[1])  # assume W:(K,N)
            gW = cp.empty_like(W)
            gB = cp.empty((1, units), dtype=cp.float32) if getattr(lyr, "b", None) is not None else None
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
    def _plan_conv2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """Conv2D 플래너.

        - 입력:  (N, Cin, H, W)
        - 출력:  (N, Cout, H_out, W_out)
        - z:     act!='none'일 때만 별도 버퍼
        - gW:    레이어의 W shape와 동일(있으면), 없으면 추정 shape
        - gB:    bias 사용 시 (Cout,)
        - work:  forward/backward 공용 Conv2DWorkspaces (모든 중간행렬 포함)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: Conv2D expects 4D input (N,C,H,W), got {cur}")
        N, Cin, H, W = map(int, cur)
        KH, KW = map(int, getattr(lyr, "kernel_size", (1, 1)))
        stride   = getattr(lyr, "stride", (1, 1))
        padding  = getattr(lyr, "padding", (0, 0))
        dilation = getattr(lyr, "dilation", (1, 1))
        groups   = max(1, int(getattr(lyr, "groups", 1)))
        Cout     = int(getattr(lyr, "out_channels"))

        H_out, W_out = _conv2d_out_hw(H, W, KH, KW, stride, padding, dilation)
        out_shp = (N, Cout, H_out, W_out)
        y = cp.empty(out_shp, dtype=cp.float32)

        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "none")
        need_z = (str(act_name).lower() != "none")
        z = (cp.empty_like(y) if need_z else None)

        gA = cp.empty((N, Cin, H, W), dtype=cp.float32)

        Wp = getattr(lyr, "W", None)
        if Wp is not None:
            gW = cp.empty_like(Wp)
        else:
            gW = cp.empty((Cout, Cin // groups, KH, KW), dtype=cp.float32)

        with_bias = bool(getattr(lyr, "use_bias", False)) or \
                    (getattr(lyr, "B", None) is not None) or \
                    (getattr(lyr, "bias", None) is not None) or \
                    (getattr(lyr, "b", None) is not None)
        gB = cp.empty((Cout,), dtype=cp.float32) if with_bias else None

        from graph_executor_v2.ops.conv2d import Conv2DWorkspaces  # type: ignore
        ws = Conv2DWorkspaces()
        HWo = H_out * W_out
        K   = (Cin // groups) * KH * KW
        # forward
        ws.dCol   = cp.empty((HWo, K),     dtype=cp.float32)
        ws.W_KC   = cp.empty((K,   Cout),  dtype=cp.float32)
        ws.Y_tmp  = cp.empty((HWo, Cout),  dtype=cp.float32)
        ws.Z_rows = cp.empty((HWo, Cout),  dtype=cp.float32) if need_z else None
        # backward
        ws.dCol_b  = cp.empty((HWo, K), dtype=cp.float32)
        ws.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
        ws.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
        ws.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
        ws.W_CK   = cp.empty((Cout, K), dtype=cp.float32)
        ws.dY_HT  = cp.empty((HWo,  Cout), dtype=cp.float32)
        ws.dWpack = cp.empty((Cout, K), dtype=cp.float32)

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 3) Pool2D
try:
    from graph_executor_v2.layers.pool2d import Pool2D as _Pool2DLayer  # type: ignore
except Exception:
    _Pool2DLayer = None  # type: ignore

if _Pool2DLayer is not None:
    @register_planner(_Pool2DLayer)  # type: ignore[misc]
    def _plan_pool2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """Pool2D 플래너.

        - max pool인 경우, backward를 위해 인덱스 버퍼(work.indices) 필요
        - y:  (N, C, Ho, Wo) / gA: (N, C, H, W) / gW/gB: 없음
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: Pool2D expects 4D input (N,C,H,W), got {cur}")
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
        out_shp = (N, C, Ho, Wo)

        y = cp.empty(out_shp, dtype=cp.float32)
        z = None
        gA = cp.empty((N, C, H, W), dtype=cp.float32)
        gW = None
        gB = None

        work = None
        if mode == "max":
            from graph_executor_v2.layers.pool2d import _Pool2DWork  # local helper
            work = _Pool2DWork()
            work.indices = cp.empty((N, C, Ho, Wo), dtype=cp.int32)

        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=work), out_shp

# 4) Activation (cupy only, shape-preserving)
try:
    from graph_executor_v2.layers.activations import ActivationLayer  # type: ignore
except Exception:
    ActivationLayer = None  # type: ignore

if ActivationLayer is not None:
    @register_planner(ActivationLayer)  # type: ignore[misc]
    def _plan_activation(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """Shape-preserving activation(별도 파라미터 없음)."""
        lname = f"L{idx}:{lyr.__class__.__name__}"
        out_shp = tuple(map(int, cur))
        y  = cp.empty(out_shp, dtype=cp.float32)
        z  = None
        gA = cp.empty(cur, dtype=cp.float32)
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
    def _plan_pad(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """Pad 레이어(파라미터 없음, shape 변경)."""
        lname = f"L{idx}:{lyr.__class__.__name__}"
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

        y = cp.empty(out_shp, dtype=cp.float32)
        z = None
        gA = cp.empty(cur, dtype=cp.float32)
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
    def _plan_bn2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """BatchNorm2d 플래너.

        - forward: y[cur], z=None
        - backward: gA[cur], gW[gamma]/gB[beta] (affine=True일 때)
        - X_saved는 graph_exec에서 prev_y로 전달(BN 제약)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: BN2d expects 4D input, got {cur}")
        out_shp = tuple(map(int, cur))

        y = cp.empty(out_shp, dtype=cp.float32)
        z = None
        gA = cp.empty(cur, dtype=cp.float32)
        if getattr(lyr, "affine", True):
            C = cur[3] if getattr(lyr, "channels_last", False) else cur[1]
            gW = cp.empty((C,), dtype=cp.float32)  # dgamma
            gB = cp.empty((C,), dtype=cp.float32)  # dbeta
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
    def _plan_embedding(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """Embedding 플래너.

        - 입력: int32 indices (N,L) 또는 (L,)
        - 출력: float32 (N,L,D) 또는 (L,D)
        - backward: gA dummy, gW(V,D), gB 없음
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

        y = cp.empty(out_shp, dtype=cp.float32)
        z = None
        gA = cp.empty(cur, dtype=cp.float32)   # dummy grad carrier
        gW = cp.empty((V, D), dtype=cp.float32)
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
        """Dropout용 내부 work: mask 버퍼 + counter_base(seed 변조)."""
        __slots__ = ("mask", "counter_base")
        def __init__(self, shape):
            self.mask = cp.empty(shape, dtype=cp.int32)  # fwd에서 채워지고 bwd에서 재사용
            self.counter_base = 0                        # replay마다 증가시켜 다른 마스크 생성

    @register_planner(_DropLayer)  # type: ignore[misc]
    def _plan_dropout(lyr, cur, idx, lt_bytes):
        lname = f"L{idx}:{lyr.__class__.__name__}"
        out_shp = tuple(map(int, cur))  # shape 보존형 레이어

        y  = cp.empty(out_shp, dtype=cp.float32)
        z  = None
        gA = cp.empty(cur, dtype=cp.float32)
        gW = None
        gB = None

        ws = _DropoutWork(out_shp)
        return PerLayerBufs(name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws), out_shp

# 9) RNN (Elman)
try:
    from graph_executor_v2.layers.rnn import RNN  # type: ignore
except Exception:
    RNN = None  # type: ignore

if RNN is not None:
    @register_planner(RNN)  # type: ignore[misc]
    def _plan_rnn(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
        """RNN(Elman) 플래너.

        - 입력: X (N,T,I)
        - 출력: y (N,T,H), z(optional) = pre-activation
        - backward: dX(N,T,I), gWx(I,H), gWh(H,H), gB(H,?) (with_bias), dh0(N,H)
        - work: RnnWorkspaces (fwd/bwd 공용 버퍼)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"

        if len(cur) != 3:
            raise RuntimeError(f"{lname}: RNN expects 3D input (N,T,I), got {cur}")
        N, T, I = map(int, cur)
        H = int(getattr(lyr, "hidden_size"))

        out_shp = (N, T, H)
        y = cp.empty(out_shp, dtype=cp.float32)

        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "tanh")
        need_z = (str(act_name).lower() != "none")
        z = (cp.empty_like(y) if need_z else None)

        gA  = cp.empty((N, T, I), dtype=cp.float32)      # dX
        gW  = None
        gWx = cp.empty((I, H),     dtype=cp.float32)
        gWh = cp.empty((H, H),     dtype=cp.float32)

        with_bias = bool(getattr(lyr, "with_bias", False)) or \
                    (getattr(lyr, "b", None) is not None) or \
                    (getattr(lyr, "bias", None) is not None)
        gB  = cp.empty((H,), dtype=cp.float32) if with_bias else None
        dh0 = cp.empty((N, H), dtype=cp.float32)

        from graph_executor_v2.ops.rnn import RnnWorkspaces  # type: ignore
        ws = RnnWorkspaces()
        # forward workspaces
        ws.XH_cat   = cp.empty((N, I+H), dtype=cp.float32)
        ws.Y_rows   = cp.empty((N, H),   dtype=cp.float32)
        ws.W_cat    = cp.empty((I+H, H), dtype=cp.float32)
        ws.Z_rows_f = cp.empty((N, H),   dtype=cp.float32) if need_z else None
        # backward workspaces
        ws.XH_cat_b = cp.empty((N, I+H), dtype=cp.float32)
        ws.G_rows   = cp.empty((N, H),   dtype=cp.float32)
        ws.Z_rows_b = cp.empty((N, H),   dtype=cp.float32)
        ws.W_cat_b  = cp.empty((I+H, H), dtype=cp.float32)
        ws.dXH_cat  = cp.empty((N, I+H), dtype=cp.float32)
        ws.dWcat    = cp.empty((I+H, H), dtype=cp.float32)
        ws.TmpW     = cp.empty((I+H, H), dtype=cp.float32)

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
        # (선택) 레이어 개별 build 지원
        if not getattr(lyr, "built", False) and hasattr(lyr, "build"):
            try:
                lyr.build(cur)
            except Exception:
                pass  # 일부 레이어는 build가 shape-independent → 무시

        planner = _find_planner(lyr)
        if planner is None:
            per, out_shp = _generic_paramless_planner(lyr, cur, idx, lt_bytes)
        else:
            per, out_shp = planner(lyr, cur, idx, lt_bytes)
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
    lt_bytes: int = (8 << 20),
) -> CapturePlan:
    """정적 경로용: Sequential 전체 모델에 대한 CapturePlan 생성.

    흐름:
      1) (필요 시) model.build(in_shape)
      2) _plan_over_layers(model.layers, in_shape)
      3) loss_kind에 따라 dY 버퍼 준비(softmax_ce만 dY 준비)
      4) CapturePlan 반환

    참고:
      - 이 단계는 CUDA Graph 캡처가 아니라 '버퍼/WS 사전할당'만 담당한다.
      - (TODO) 이 Plan을 ExecPlanner에 전달하여 DAG/스케줄 메타를 만들 수 있다.
    """
    cur = tuple(map(int, input_shape))

    # 필요 시 자동 build
    if not getattr(model, "built", False):
        if hasattr(model, "build"):
            model.build(cur)

    layers = list(getattr(model, "layers", []))
    per_layer, out_shape = _plan_over_layers(layers, cur, lt_bytes=lt_bytes)

    # loss dY (현재 softmax_ce만 별도 dY 운용)
    dY = cp.empty(out_shape, dtype=cp.float32) if loss_kind == "softmax_ce" else None

    return CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(out_shape)),
    )

def make_plan_for_path(
    path_layers: Sequence[Any],
    input_shape: Tuple[int, ...],
    *,
    loss_kind: str = "softmax_ce",
    lt_bytes: int = (8 << 20),
) -> CapturePlan:
    """동적 경로용: 평탄화된 '선형 경로' 레이어 리스트에 대한 CapturePlan 생성.

    - Sequential 컨테이너를 거치지 않고도 직접 사용 가능(then/else 블록 등)
    - 내부 동작은 make_plan_for_sequential과 동일하되, 대상 레이어 시퀀스만 다름
    """
    cur = tuple(map(int, input_shape))
    per_layer, out_shape = _plan_over_layers(path_layers, cur, lt_bytes=lt_bytes)

    dY = cp.empty(out_shape, dtype=cp.float32) if loss_kind == "softmax_ce" else None

    return CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(out_shape)),
    )

# ============================================================
# Utilities
# ============================================================

def advance_dropout(plan: CapturePlan, seed_bump: int = 1) -> None:
    """Dropout 마스크 재생성 유도(경로/스텝 간 변동성 확보).

    - Replay 루프에서 매 스텝마다 호출할 수 있도록 분리.
    - 각 레이어의 work.counter_base를 증가시켜 PRNG offset에 반영.s
    - 정책적으로 꺼두고 싶으면 seed_bump=0 사용.
    """
    if seed_bump == 0:
        return
    for per in plan.per_layer:
        work = getattr(per, "work", None)
        if work is not None and hasattr(work, "counter_base"):
            try:
                work.counter_base += int(seed_bump)
            except Exception:
                pass
