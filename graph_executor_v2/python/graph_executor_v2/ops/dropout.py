# python/graph_executor_v2/ops/dropout.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_dropout as _g
except Exception as e:
    raise ImportError(
        "[ops.dropout] _ops_dropout 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_dropout)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


def _assert_f32_2d(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray expected")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 required")
    if x.ndim != 2:
        raise ValueError(f"{name}: 2D [M,N] required")


def _mk_attrs(p: float, seed: int, scale_in_train: bool, counter_base: int):
    a = _g.DropoutAttrs()
    a.p = float(p)
    a.seed = int(seed) & ((1<<64)-1)
    a.scale_in_train = bool(scale_in_train)
    a.counter_base = int(counter_base) & ((1<<64)-1)
    return a


def forward(
    X: cp.ndarray,
    *,
    p: float = 0.1,
    seed: int = 0x1234,
    counter_base: int = 0,
    scale_in_train: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
    out_mask: Optional[cp.ndarray] = None,   # int32 [M,N] or None
):
    _assert_f32_2d(X, "X")
    M, N = map(int, X.shape)
    Xc = cp.ascontiguousarray(X)

    if out is None:
        out = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != X.shape:
            raise ValueError("out shape must equal X shape")
        out = cp.ascontiguousarray(out)

    mask_ptr = None
    mask_shape = []
    if out_mask is not None:
        if not isinstance(out_mask, cp.ndarray) or out_mask.dtype != cp.int32 or out_mask.ndim != 2 or out_mask.shape != X.shape:
            raise TypeError("out_mask must be int32 [M,N] matching X")
        out_mask = cp.ascontiguousarray(out_mask)
        mask_ptr = int(out_mask.data.ptr)
        mask_shape = [M, N]

    attrs = _mk_attrs(p, seed, scale_in_train, counter_base)
    sptr = int(get_stream_ptr(stream))
    _g.forward(
        int(Xc.data.ptr), [M, N],
        int(out.data.ptr), [M, N],
        mask_ptr if mask_ptr is not None else None, mask_shape,
        attrs, sptr
    )
    return out, out_mask


def backward(
    dY: cp.ndarray,
    mask: cp.ndarray,
    *,
    p: float = 0.1,
    scale_in_train: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
):
    _assert_f32_2d(dY, "dY")
    if not (isinstance(mask, cp.ndarray) and mask.dtype == cp.int32 and mask.ndim == 2 and mask.shape == dY.shape):
        raise TypeError("mask must be int32 [M,N] matching dY")

    M, N = map(int, dY.shape)
    dYc = cp.ascontiguousarray(dY)
    maskc = cp.ascontiguousarray(mask)

    if out is None:
        out = cp.empty_like(dYc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != dY.shape:
            raise ValueError("out shape must equal dY shape")
        out = cp.ascontiguousarray(out)

    attrs = _mk_attrs(p, seed=0, scale_in_train=scale_in_train, counter_base=0)
    sptr = int(get_stream_ptr(stream))
    _g.backward(
        int(dYc.data.ptr), [M, N],
        int(maskc.data.ptr), [M, N],
        int(out.data.ptr), [M, N],
        attrs, sptr
    )
    return out
