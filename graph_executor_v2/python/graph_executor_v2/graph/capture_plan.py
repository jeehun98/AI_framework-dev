from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
import cupy as cp

__all__ = [
    "PerLayerBufs", "LossBufs", "CapturePlan",
    "register_planner", "make_plan_for_sequential",
]

# ============================================================
# Core dataclasses
# ============================================================

@dataclass
class PerLayerBufs:
    name: str
    # forward
    y: cp.ndarray                        # forward output (required)
    z: Optional[cp.ndarray]              # pre-activation (optional)
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
# Planner registry
# ============================================================

_Planner = Callable[
    [Any, Tuple[int, ...], int, int],  # (layer, cur_in_shape, layer_index, lt_bytes)
    Tuple[PerLayerBufs, Tuple[int, ...]]  # (per_layer_bufs, out_shape)
]
_PLANNERS: Dict[Type[Any], _Planner] = {}

def register_planner(layer_type: Type[Any]):
    def _wrap(func: _Planner):
        _PLANNERS[layer_type] = func
        return func
    return _wrap

def _find_planner(lyr: Any) -> Optional[_Planner]:
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
        lname = f"L{idx}:{lyr.__class__.__name__}"
        try:
            out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
        except Exception as e:
            raise RuntimeError(f"cannot infer output shape for {lname}: {e}")

        y = cp.empty(out_shp, dtype=cp.float32)
        need_z = getattr(lyr, "activation", "none") != "none"
        z = cp.empty(out_shp, dtype=cp.float32) if need_z else None

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

# 2) Conv2D
try:
    from graph_executor_v2.layers.conv2d import Conv2D  # type: ignore
except Exception:
    Conv2D = None  # type: ignore

if Conv2D is not None:
    @register_planner(Conv2D)  # type: ignore[misc]
    def _plan_conv2d(lyr, cur: Tuple[int, ...], idx: int, lt_bytes: int):
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
        z = y if not need_z else cp.empty_like(y)

        gA = cp.empty((N, Cin, H, W), dtype=cp.float32)
        gW = cp.empty((Cout, Cin, KH, KW), dtype=cp.float32)
        with_bias = (
            bool(getattr(lyr, "with_bias", False)) or
            (getattr(lyr, "B", None) is not None) or
            (getattr(lyr, "bias", None) is not None)
        )
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
        """
        Pad 레이어 플래너:
          - forward: y[out_shape], z=None
          - backward: gA[cur], no params
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        # 레이어가 compute_output_shape를 제공한다고 가정
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
        """
        BatchNorm2d 플래너:
          - forward: y[cur], z=None
          - backward: gA[cur], gW[gamma] / gB[beta] (affine일 때)
          - work: None (X_saved는 graph_exec에서 이전 레이어 y로 전달)
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"
        if len(cur) != 4:
            raise RuntimeError(f"{lname}: BN2d expects 4D input, got {cur}")
        out_shp = tuple(map(int, cur))

        y = cp.empty(out_shp, dtype=cp.float32)
        z = None
        gA = cp.empty(cur, dtype=cp.float32)
        if getattr(lyr, "affine", True):
            # gamma/beta shape: [C]
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
        """
        Embedding 플래너:
          - 입력: int32 indices, shape = (N, L) 또는 (L,)
          - 출력: float32, shape = (N, L, D) 또는 (L, D)  (D=embedding_dim)
          - backward:
              * gA: 상류 체인용 dummy(float32) 버퍼 (입력과 동일 shape). Embedding은 입력 grad가 없으므로 0으로 채움.
              * gW: (V, D)  (V=num_embeddings)
              * gB: 없음
          - work: 없음
        """
        lname = f"L{idx}:{lyr.__class__.__name__}"

        # 입력 shape(cur) 검증 및 출력 shape 계산
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

        # forward 버퍼
        y = cp.empty(out_shp, dtype=cp.float32)
        z = None  # pre-activation 개념 없음

        # backward 버퍼
        #  - gA: 입력과 동일 shape(float32) — 그래프 체인 정합(Embedding은 실제 grad 없음)
        gA = cp.empty(cur, dtype=cp.float32)
        #  - gW: 파라미터 grad (V,D)
        gW = cp.empty((V, D), dtype=cp.float32)
        gB = None
        ws = None

        return PerLayerBufs(
            name=lname, y=y, z=z, gA=gA, gW=gW, gB=gB, work=ws
        ), tuple(out_shp)

# 8) Dropout (mask만 마련하는 소형 work)
try:
    from graph_executor_v2.layers.dropout import Dropout as _DropLayer  # type: ignore
except Exception:
    _DropLayer = None  # type: ignore

if _DropLayer is not None:
    class _DropoutWork:
        __slots__ = ("mask", "counter_base")
        def __init__(self, shape):
            self.mask = cp.empty(shape, dtype=cp.int32)  # fwd에서 채워지고 bwd에서 재사용
            self.counter_base = 0                        # 필요하면 외부에서 증가시켜 다른 마스크 생성

    @register_planner(_DropLayer)  # type: ignore[misc]
    def _plan_dropout(lyr, cur, idx, lt_bytes):
        lname = f"L{idx}:{lyr.__class__.__name__}"
        out_shp = tuple(map(int, cur))  # shape 보존형 레이어

        y  = cp.empty(out_shp, dtype=cp.float32)
        z  = None
        gA = cp.empty(cur, dtype=cp.float32)  # 입력 grad 버퍼(항등 or dropout bwd 결과)
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
        """
        RNN(Elman) 플래너:
          - 입력: X (N,T,I)
          - 출력: y (N,T,H), z(optional) = pre-activation (act=='none'이면 y와 alias)
          - backward 버퍼:
              * gA: dX (N,T,I)
              * gWx: (I,H), gWh: (H,H), gB: (H,) (with_bias가 True일 때)
              * dh0: (N,H)
          - work: RnnWorkspaces (forward/backward 공용 버퍼)
        """
        import cupy as cp
        lname = f"L{idx}:{lyr.__class__.__name__}"

        # 입력 형태 확인
        if len(cur) != 3:
            raise RuntimeError(f"{lname}: RNN expects 3D input (N,T,I), got {cur}")
        N, T, I = map(int, cur)
        H = int(getattr(lyr, "hidden_size"))

        # 출력/보조 버퍼
        out_shp = (N, T, H)
        y = cp.empty(out_shp, dtype=cp.float32)

        act_name = getattr(lyr, "activation", None) or getattr(lyr, "act", "tanh")
        need_z = (str(act_name).lower() != "none")
        z = y if not need_z else cp.empty_like(y)

        # grads
        gA  = cp.empty((N, T, I), dtype=cp.float32)      # dX
        gW  = None                                       # 단일 weight grad는 사용 안 함 (RNN은 Wx/Wh로 분리)
        gWx = cp.empty((I, H),     dtype=cp.float32)     # dWx
        gWh = cp.empty((H, H),     dtype=cp.float32)     # dWh

        with_bias = bool(getattr(lyr, "with_bias", False)) or \
                    (getattr(lyr, "b", None) is not None) or \
                    (getattr(lyr, "bias", None) is not None)
        gB  = cp.empty((H,), dtype=cp.float32) if with_bias else None
        dh0 = cp.empty((N, H), dtype=cp.float32)         # ∂L/∂h0

        # 워크스페이스
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

    # 필요 시 자동 build
    if not getattr(model, "built", False):
        if hasattr(model, "build"):
            model.build(cur)

    per_layer: List[PerLayerBufs] = []
    for idx, lyr in enumerate(getattr(model, "layers", [])):
        planner = _find_planner(lyr)
        if planner is None:
            per, out_shp = _generic_paramless_planner(lyr, cur, idx, lt_bytes)
        else:
            per, out_shp = planner(lyr, cur, idx, lt_bytes)
        per_layer.append(per)
        cur = tuple(map(int, out_shp))

    # loss dY
    dY = cp.empty(cur, dtype=cp.float32) if loss_kind == "softmax_ce" else None

    return CapturePlan(
        input_shape=tuple(map(int, input_shape)),
        per_layer=per_layer,
        loss=LossBufs(dY=dY, out_shape=tuple(cur)),
    )
