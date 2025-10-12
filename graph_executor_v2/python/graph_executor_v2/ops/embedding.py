# python/graph_executor_v2/ops/embedding.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_embedding as _g
except Exception as e:
    raise ImportError("[ops.embedding] _ops_embedding 바인딩을 찾을 수 없습니다.") from e

def _assert_i_nd(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray")
    if x.dtype != cp.int32:  # I64 미지원
        raise TypeError(f"{name}: expected int32 (I64 not supported), got {x.dtype}")
    if x.ndim not in (1,2):
        raise ValueError(f"{name}: expected 1D or 2D")
    
def _assert_f2(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray) or x.dtype!=cp.float32 or x.ndim!=2:
        raise TypeError(f"{name}: expected float32 [*,*]")

def _assert_f3_or_f2(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray) or x.dtype!=cp.float32 or x.ndim not in (2,3):
        raise TypeError(f"{name}: expected float32 rank 2 or 3")

def _attrs(padding_idx: int=-1, scale_grad_by_freq: bool=False, out_scale: float=1.0):
    a = _g.EmbeddingAttrs()
    a.padding_idx = int(padding_idx)
    a.scale_grad_by_freq = bool(scale_grad_by_freq)
    a.out_scale = float(out_scale)
    return a

def forward(
    W: cp.ndarray,           # [V,D] float32
    I: cp.ndarray,           # [N,L] or [L] int32/int64
    *,
    padding_idx: int = -1,
    out_scale: float = 1.0,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    _assert_f2(W, "W"); _assert_i_nd(I, "I")
    V, D = map(int, W.shape)
    if I.ndim==2:
        N, L = map(int, I.shape)
        y_shape = (N, L, D)
    else:
        L = int(I.shape[0]); y_shape = (L, D)
    if out is None:
        out = cp.empty(y_shape, dtype=cp.float32)
    else:
        if out.dtype!=cp.float32 or tuple(out.shape)!=y_shape:
            raise ValueError(f"out must be float32 {y_shape}")

    a = _attrs(padding_idx=padding_idx, out_scale=out_scale)
    sptr = int(get_stream_ptr(stream))
    _g.forward(
        int(W.data.ptr), list(W.shape),
        int(I.data.ptr), list(I.shape),
        int(out.data.ptr), list(out.shape),
        a, sptr
    )
    return out

def backward(
    I: cp.ndarray,           # [N,L] or [L]
    dY: cp.ndarray,          # [N,L,D] or [L,D]
    *,
    V: int, D: int,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    stream: Optional[int] = None,
    dW_out: Optional[cp.ndarray] = None
) -> Optional[cp.ndarray]:
    _assert_i_nd(I, "I"); _assert_f3_or_f2(dY, "dY")
    if dW_out is None:
        dW_out = cp.zeros((V, D), dtype=cp.float32)
    else:
        if dW_out.dtype!=cp.float32 or tuple(dW_out.shape)!=(V,D):
            raise ValueError(f"dW_out must be float32 {(V,D)}")
    a = _attrs(padding_idx=padding_idx, scale_grad_by_freq=scale_grad_by_freq)
    sptr = int(get_stream_ptr(stream))
    _g.backward(
        int(I.data.ptr), list(I.shape),
        int(dY.data.ptr), list(dY.shape),
        int(dW_out.data.ptr), [V, D],
        a, sptr
    )
    return dW_out
