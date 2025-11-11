# python/graph_executor_v2/ops/gemm.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, Union
import cupy as cp

from .common import (
    assert_f32_2d,
    as_tensor_2d,
    empty_like_2d,
    empty_2d,
    get_stream_ptr,
    ensure_cuda_dlls,
    to_voidp_capsule,
)

from graph_executor_v2.ops import _ops_gemm as g

ensure_cuda_dlls()

# ------------------------------- Strict stream helper -------------------------------
def _get_stream_ptr_strict(stream: Optional[int]) -> int:
    """
    get_stream_ptr() 포장: legacy/default(=0) 스트림 금지.
    명시 포인터 또는 현재 CuPy 스트림 포인터만 허용.
    """
    ptr = int(get_stream_ptr(stream))
    if ptr == 0:
        raise RuntimeError("Forbidden default/legacy stream (ptr==0)")
    return ptr

# ------------------------------- Workspace helpers -------------------------------
class GemmWorkspaces:
    def __init__(self):
        self.dZ: Optional[cp.ndarray] = None
        self.lt_ws: Optional[cp.ndarray] = None

    def ensure_backward(self, M: int, N: int):
        if self.dZ is None or self.dZ.dtype != cp.float32 or tuple(self.dZ.shape) != (M, N):
            raise ValueError(f"[capture] work.dZ must be preallocated float32[{M},{N}]")
        if self.lt_ws is not None and self.lt_ws.dtype != cp.uint8:
            raise ValueError("[capture] work.lt_ws must be uint8 or None")

_WS_CACHE: Dict[Tuple[int,int,int], GemmWorkspaces] = {}

def ensure_workspaces(M: int, N: int, *, lt_bytes: int = 0) -> GemmWorkspaces:
    dev = int(cp.cuda.runtime.getDevice())
    key = (M, N, dev)
    ws = _WS_CACHE.get(key)
    if ws is None:
        ws = GemmWorkspaces()
        ws.dZ = cp.empty((M, N), dtype=cp.float32)
        ws.lt_ws = cp.empty(lt_bytes, dtype=cp.uint8) if lt_bytes > 0 else None
        _WS_CACHE[key] = ws
    else:
        if ws.dZ is None or ws.dZ.shape != (M, N) or ws.dZ.dtype != cp.float32:
            ws.dZ = cp.empty((M, N), dtype=cp.float32)
        if lt_bytes > 0 and (ws.lt_ws is None or ws.lt_ws.nbytes < lt_bytes or ws.lt_ws.dtype != cp.uint8):
            ws.lt_ws = cp.empty(lt_bytes, dtype=cp.uint8)
    return ws

def clear_ws_cache() -> None:
    _WS_CACHE.clear()

# ------------------------------- Attr helpers -------------------------------
def _parse_act_to_kind(act: str) -> "g.ActKind":
    s = (act or "none").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if s == "none":    return getattr(g.ActKind, "None")
    if s == "relu":    return g.ActKind.ReLU
    if s in ("leakyrelu","lrelu"): return g.ActKind.LeakyReLU
    if s == "gelu":    return g.ActKind.GELU
    if s == "sigmoid": return g.ActKind.Sigmoid
    if s == "tanh":    return g.ActKind.Tanh
    raise ValueError(f"unknown act: {act}")

def _mk_attrs(act: str, with_bias: bool, leaky_slope: float) -> "g.GemmAttrs":
    a = g.GemmAttrs()
    a.trans_a = False
    a.trans_b = False
    a.act = _parse_act_to_kind(act)
    a.with_bias = bool(with_bias)
    a.leaky_slope = float(leaky_slope)
    return a

def _has(name: str) -> bool:
    return hasattr(g, name)

# ============================================================
# Forward (allocating)
# ============================================================
def forward(
    A: cp.ndarray,
    B: cp.ndarray,
    bias: Optional[cp.ndarray] = None,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
    save_z: bool = False,
    z_out: Optional[cp.ndarray] = None,
    return_z: bool = False,
) -> Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")

    if bias is not None and not with_bias:
        with_bias = True
    if bias is not None and with_bias is False:
        raise ValueError("[capture] bias provided but with_bias=False ...")

    tBias = None
    if with_bias:
        if bias is None:
            raise ValueError("with_bias=True but bias is None")
        if cp.isscalar(bias):
            bias = cp.asarray(bias, dtype=cp.float32).reshape(1, 1)
        elif isinstance(bias, cp.ndarray):
            if bias.dtype != cp.float32:
                bias = bias.astype(cp.float32, copy=False)
            if bias.ndim == 1:
                if bias.size != N:
                    raise ValueError(f"bias 1D length must be N={N}, got {bias.size}")
                bias = bias.reshape(1, N)
            elif bias.ndim == 2:
                bm, bn = bias.shape
                if not ((bm == 1 and bn == N) or (bm == M and bn == 1) or (bm == M and bn == N)):
                    raise ValueError(f"unsupported bias shape {bias.shape}; expected (1,N)|(M,1)|(M,N)")
            else:
                raise ValueError(f"unsupported bias ndim={bias.ndim}")
        else:
            raise TypeError("bias must be scalar or CuPy ndarray")
        tBias = as_tensor_2d(bias)

    if out is None:
        out = cp.empty((M, N), dtype=cp.float32)
    elif out.shape != (M, N) or out.dtype != cp.float32:
        raise ValueError(f"out must be float32[{M},{N}]")

    act_is_none = (act or "none").strip().lower() in ("none",)

    if z_out is not None:
        if z_out.shape != (M, N) or z_out.dtype != cp.float32:
            raise ValueError(f"z_out must be float32[{M},{N}]")
        tZ_np = z_out
    elif act_is_none:
        tZ_np = out  # alias
    elif save_z or return_z:
        tZ_np = cp.empty((M, N), dtype=cp.float32)
    else:
        tZ_np = None

    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(out)
    tZ = as_tensor_2d(tZ_np) if tZ_np is not None else None
    stream_ptr = _get_stream_ptr_strict(stream)

    if _has("forward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        try:
            attrs.save_z = bool(tZ is not None)
        except Exception:
            pass
        g.forward_ex_attrs(tA, tB, tBias, tY, attrs, tZ, to_voidp_capsule(stream_ptr))
    else:
        g.forward_ex(
            tA, tB, tBias, tY,
            False, False,
            act, bool(with_bias), float(leaky_slope),
            bool(tZ is not None),
            tZ,
            to_voidp_capsule(stream_ptr),
        )

    if tZ is not None and (return_z or save_z):
        return out, tZ_np
    if return_z:
        return out, (tZ_np if tZ_np is not None else out)
    return out

# ============================================================
# Backward (allocating)
# ============================================================
def backward(
    A: cp.ndarray,
    B: cp.ndarray,
    gY: cp.ndarray,
    Z: cp.ndarray,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    C: Optional[cp.ndarray] = None,
    want_gA: bool = True,
    want_gB: bool = True,
    want_gBias: bool = False,
    stream: Optional[int] = None,
    warn_mismatch: bool = False,
) -> Dict[str, cp.ndarray]:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")
    Mg, Ng = assert_f32_2d(gY, "gY")
    Mz, Nz = assert_f32_2d(Z,  "Z")
    if (Mg, Ng) != (M, N) or (Mz, Nz) != (M, N):
        raise ValueError(f"Shape mismatch: gY(Z) must be (M={M}, N={N})")

    if not isinstance(with_bias, bool):
        with_bias = bool(with_bias)
    if with_bias is False and want_gBias:
        raise ValueError("want_gBias=True requires with_bias=True")

    tA  = as_tensor_2d(A)
    tB  = as_tensor_2d(B)
    tgY = as_tensor_2d(gY)
    tZ  = as_tensor_2d(Z)
    tC  = as_tensor_2d(C) if C is not None else None

    gA_arr = empty_like_2d(A) if want_gA else None
    gB_arr = empty_like_2d(B) if want_gB else None
    gC_arr = empty_like_2d(Z) if (C is not None) else None
    gBias_arr = empty_2d(1, N) if (want_gBias and with_bias) else None

    t_gA    = as_tensor_2d(gA_arr)    if gA_arr    is not None else None
    t_gB    = as_tensor_2d(gB_arr)    if gB_arr    is not None else None
    t_gC    = as_tensor_2d(gC_arr)    if gC_arr    is not None else None
    t_gBias = as_tensor_2d(gBias_arr) if gBias_arr is not None else None

    stream_ptr = _get_stream_ptr_strict(stream)

    if _has("backward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        g.backward_ex_attrs(tA, tB, tC, tgY, tZ, t_gA, t_gB, t_gC, t_gBias, attrs, to_voidp_capsule(stream_ptr))
    else:
        g.backward_ex(
            tA, tB, tC, tgY, tZ,
            t_gA, t_gB, t_gC, t_gBias,
            False, False,
            act, bool(with_bias), float(leaky_slope),
            to_voidp_capsule(stream_ptr),
        )

    out: Dict[str, cp.ndarray] = {}
    if gA_arr is not None:    out["gA"]    = gA_arr
    if gB_arr is not None:    out["gB"]    = gB_arr
    if gC_arr is not None:    out["gC"]    = gC_arr
    if gBias_arr is not None: out["gBias"] = gBias_arr
    return out

# ============================================================
# Forward (capture-safe, NO allocations)
# ============================================================
def forward_into(
    A: cp.ndarray,
    B: cp.ndarray,
    *,
    out: cp.ndarray,
    bias: Optional[cp.ndarray] = None,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    save_z: bool = False,
    z_out: Optional[cp.ndarray] = None,
    stream: Optional[int] = None,
) -> None:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")

    if out is None or out.dtype != cp.float32 or out.shape != (M, N) or not out.flags.c_contiguous:
        raise ValueError(f"[capture] out must be C-contiguous float32[{M},{N}]")
    if not (A.flags.c_contiguous and B.flags.c_contiguous):
        raise ValueError("[capture] inputs must be C-contiguous")

    if bias is not None and with_bias is False:
        raise ValueError("[capture] bias provided but with_bias=False")

    tBias = None
    if with_bias:
        if bias is None:
            raise ValueError("[capture] with_bias=True but bias is None")
        if not (isinstance(bias, cp.ndarray) and bias.dtype == cp.float32 and bias.ndim == 2 and bias.shape == (1, N) and bias.flags.c_contiguous):
            raise ValueError("[capture] bias must be C-contiguous float32[1,N] (PerN)")
        tBias = as_tensor_2d(bias)

    act_is_none = (act or "none").strip().lower() in ("none",)

    # ❗ NO-alloc 정책: save_z가 필요한데 z_out이 없으면 실패
    if z_out is not None:
        if z_out.dtype != cp.float32 or z_out.shape != (M, N) or not z_out.flags.c_contiguous:
            raise ValueError(f"[capture] z_out must be C-contiguous float32[{M},{N}]")
        tZ_np = z_out
    elif act_is_none:
        tZ_np = out
    elif save_z:
        raise ValueError("[capture] save_z=True requires z_out preallocated")
    else:
        tZ_np = None

    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(out)
    tZ = as_tensor_2d(tZ_np) if tZ_np is not None else None
    stream_ptr = _get_stream_ptr_strict(stream)

    if _has("forward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        try:
            attrs.save_z = bool(tZ is not None)
        except Exception:
            pass
        g.forward_ex_attrs(tA, tB, tBias, tY, attrs, tZ, to_voidp_capsule(stream_ptr))
    else:
        g.forward_ex(
            tA, tB, tBias, tY,
            False, False,
            act, bool(with_bias), float(leaky_slope),
            bool(tZ is not None),
            tZ,
            to_voidp_capsule(stream_ptr),
        )

# ============================================================
# Backward (capture-safe, NO allocations)
# ============================================================
def backward_into(
    A: cp.ndarray,
    B: cp.ndarray,
    gY: cp.ndarray,
    Z: cp.ndarray,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    C: Optional[cp.ndarray] = None,
    gA_out: Optional[cp.ndarray] = None,
    gB_out: Optional[cp.ndarray] = None,
    gC_out: Optional[cp.ndarray] = None,
    gBias_out: Optional[cp.ndarray] = None,
    stream: Optional[int] = None,
    work_dZ: Optional[cp.ndarray] = None,
    lt_workspace: Optional[cp.ndarray] = None,
) -> None:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")
    Mg, Ng = assert_f32_2d(gY, "gY")
    Mz, Nz = assert_f32_2d(Z,  "Z")
    if (Mg, Ng) != (M, N) or (Mz, Nz) != (M, N):
        raise ValueError(f"[capture] gY/Z must be (M={M}, N={N})")

    def _chk(arr, shape, name):
        if arr is None: return
        if arr.dtype != cp.float32 or tuple(arr.shape) != shape or not arr.flags.c_contiguous:
            raise ValueError(f"[capture] {name} must be C-contiguous float32{shape}")
    _chk(gA_out, (M, K), "gA_out")
    _chk(gB_out, (K, N), "gB_out")

    if C is None:
        if gC_out is not None:
            raise ValueError("[capture] gC_out must be None when C is None")
    else:
        Mc, Nc = assert_f32_2d(C, "C")
        if (Mc, Nc) != (M, N):
            raise ValueError(f"[capture] C must be (M={M}, N={N})")
        _chk(gC_out, (M, N), "gC_out")

    if with_bias:
        _chk(gBias_out, (1, N), "gBias_out")
    else:
        if gBias_out is not None:
            raise ValueError("[capture] gBias_out must be None when with_bias=False")

    # ❗ NO-alloc 정책: work_dZ 미제공 시 실패
    if work_dZ is None or work_dZ.dtype != cp.float32 or work_dZ.size != M * N or not work_dZ.flags.c_contiguous:
        raise ValueError("[capture] work_dZ must be C-contiguous float32 with size M*N")
    dZ_ptr = int(work_dZ.data.ptr)

    if lt_workspace is not None:
        if lt_workspace.dtype != cp.uint8 or not lt_workspace.flags.c_contiguous:
            raise ValueError("[capture] lt_workspace must be C-contiguous uint8")
        lt_ptr   = int(lt_workspace.data.ptr)
        lt_bytes = int(lt_workspace.nbytes)
    else:
        lt_ptr, lt_bytes = 0, 0

    tA  = as_tensor_2d(A)
    tB  = as_tensor_2d(B)
    tgY = as_tensor_2d(gY)
    tZ  = as_tensor_2d(Z)
    tC  = as_tensor_2d(C) if C is not None else None
    t_gA    = as_tensor_2d(gA_out)    if gA_out    is not None else None
    t_gB    = as_tensor_2d(gB_out)    if gB_out    is not None else None
    t_gC    = as_tensor_2d(gC_out)    if gC_out    is not None else None
    t_gBias = as_tensor_2d(gBias_out) if gBias_out is not None else None

    attrs = _mk_attrs(act, with_bias, leaky_slope)
    stream_ptr = _get_stream_ptr_strict(stream)

    if not _has("backward_into"):
        raise RuntimeError("g.backward_into entrypoint not found in bindings (capture-safe BWD unavailable)")

    g.backward_into(
        tA, tB, tC, tgY, tZ,
        t_gA, t_gB, t_gC, t_gBias,
        attrs,
        to_voidp_capsule(stream_ptr),
        int(dZ_ptr),
        int(lt_ptr),
        int(lt_bytes),
    )
