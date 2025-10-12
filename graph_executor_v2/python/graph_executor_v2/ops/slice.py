# python/graph_executor_v2/ops/slice.py
from __future__ import annotations
from typing import Sequence, Optional
import cupy as cp
from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()
from graph_executor_v2.ops import _ops_slice as _g

def _assert_f32(x,name):
    if not isinstance(x,cp.ndarray) or x.dtype!=cp.float32 or not x.flags.c_contiguous:
        raise TypeError(f"{name}: need float32 C-contiguous")
    if x.ndim<1 or x.ndim>4: raise ValueError("rank must be 1..4")

def forward(X: cp.ndarray, starts: Sequence[int], sizes: Sequence[int],
            out: Optional[cp.ndarray]=None, stream: Optional[int]=None)->cp.ndarray:
    _assert_f32(X,"X"); rank=X.ndim
    if len(starts)!=rank or len(sizes)!=rank: raise ValueError("starts/sizes length mismatch")
    for d in range(rank):
        if starts[d]<0 or starts[d]+sizes[d]>X.shape[d]:
            raise ValueError("slice out of bounds")
    y_shape=tuple(sizes)
    if out is None: out = cp.empty(y_shape,dtype=cp.float32)
    else:
        if out.dtype!=cp.float32 or tuple(out.shape)!=y_shape or not out.flags.c_contiguous:
            raise ValueError("out mismatch")
    a=_g.SliceAttrs(); a.rank=rank
    for d in range(4):
        a.starts[d] = int(starts[d]) if d<rank else 0
        a.sizes[d]  = int(sizes[d])  if d<rank else 1
    _g.forward(int(X.data.ptr), list(X.shape),
               int(out.data.ptr), list(out.shape),
               a, int(get_stream_ptr(stream)))
    return out
