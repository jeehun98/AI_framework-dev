from __future__ import annotations
from typing import List, Optional
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_common as _c
    from graph_executor_v2.ops import _ops_concat as _g
except Exception as e:
    raise ImportError("[ops.concat] _ops_concat 바인딩을 찾을 수 없습니다.") from e


# --------------------------- helpers ---------------------------
def _check_f32(x: cp.ndarray, name="x"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요 (got {x.dtype})")


def _norm_axis(axis: int, ndim: int) -> int:
    if not isinstance(axis, int):
        raise TypeError("axis must be int")
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"axis out of range: {axis} for ndim={ndim}")
    return axis


# --------------------------- API ---------------------------
def forward(xs: List[cp.ndarray], axis: int, out: Optional[cp.ndarray] = None,
            stream: Optional[int] = None) -> cp.ndarray:
    """
    Concatenate xs along axis (float32 only).
    """
    if not xs:
        raise ValueError("xs is empty")
    for i, x in enumerate(xs):
        _check_f32(x, f"xs[{i}]")
        if not x.flags.c_contiguous:
            xs[i] = cp.ascontiguousarray(x)

    ndim = xs[0].ndim
    axis = _norm_axis(axis, ndim)
    base = list(xs[0].shape)

    # shape 검증
    cat_len = 0
    for x in xs:
        if x.ndim != ndim:
            raise ValueError("all inputs must have same ndim")
        for d in range(ndim):
            if d == axis:
                continue
            if x.shape[d] != base[d]:
                raise ValueError("non-concat dims must match")
        cat_len += x.shape[axis]

    out_shape = list(base)
    out_shape[axis] = cat_len

    if out is None:
        out = cp.empty(tuple(out_shape), dtype=cp.float32)
    else:
        _check_f32(out, "out")
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(f"out shape mismatch: expected {tuple(out_shape)}, got {out.shape}")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

    sptr = int(get_stream_ptr(stream))
    ptrs = [int(x.data.ptr) for x in xs]
    shapes = [list(map(int, x.shape)) for x in xs]

    _g.forward(ptrs, shapes, int(out.data.ptr), list(map(int, out_shape)), int(axis), sptr)
    return out


def backward(xs: List[cp.ndarray], axis: int, gy: cp.ndarray,
             stream: Optional[int] = None) -> List[cp.ndarray]:
    """
    Split gy back into grads for xs (float32 only).
    """
    if not xs:
        raise ValueError("xs is empty")
    for i, x in enumerate(xs):
        _check_f32(x, f"xs[{i}]")
        if not x.flags.c_contiguous:
            xs[i] = cp.ascontiguousarray(x)

    _check_f32(gy, "gy")
    if not gy.flags.c_contiguous:
        gy = cp.ascontiguousarray(gy)

    axis = _norm_axis(axis, gy.ndim)

    # out grads
    gxs = [cp.empty_like(x) for x in xs]
    sptr = int(get_stream_ptr(stream))

    x_ptrs = [int(x.data.ptr) for x in xs]
    x_shapes = [list(map(int, x.shape)) for x in xs]
    gx_ptrs = [int(gx.data.ptr) for gx in gxs]

    _g.backward(x_ptrs, x_shapes, int(axis), int(gy.data.ptr), list(map(int, gy.shape)), gx_ptrs, sptr)
    return gxs
