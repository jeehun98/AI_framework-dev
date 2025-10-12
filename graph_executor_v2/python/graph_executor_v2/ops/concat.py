# python/graph_executor_v2/ops/concat.py
from __future__ import annotations
from typing import List, Tuple, Optional
import cupy as cp
from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()
from graph_executor_v2.ops import _ops_concat as _g

def _assert_f32(x: cp.ndarray, name="arr"):
    if not isinstance(x, cp.ndarray) or x.dtype!=cp.float32:
        raise TypeError(f"{name}: expect cupy.float32")
    if not x.flags.c_contiguous:
        raise ValueError(f"{name}: must be C-contiguous")
    if x.ndim<1 or x.ndim>4:
        raise ValueError(f"{name}: rank must be 1..4")

def forward(inputs: List[cp.ndarray], axis: int = 0, out: Optional[cp.ndarray]=None, stream: Optional[int]=None)->cp.ndarray:
    if len(inputs)==0: raise ValueError("inputs empty")
    for i,t in enumerate(inputs): _assert_f32(t, f"inputs[{i}]")
    rank = inputs[0].ndim
    if any(t.ndim!=rank for t in inputs): raise ValueError("rank mismatch")
    shape_base = list(inputs[0].shape)
    if not (0<=axis<rank): raise ValueError("bad axis")
    cat_size = sum(t.shape[axis] for t in inputs)
    out_shape = shape_base[:]; out_shape[axis]=cat_size
    for t in inputs[1:]:
        for d in range(rank):
            if d==axis: continue
            if t.shape[d]!=shape_base[d]: raise ValueError("shape mismatch except axis")

    if out is None: out = cp.empty(tuple(out_shape), dtype=cp.float32)
    else:
        _assert_f32(out, "out")
        if tuple(out.shape)!=tuple(out_shape): raise ValueError("out shape mismatch")

    attrs = _g.ConcatAttrs(); attrs.axis=int(axis)
    _g.forward(
        [int(t.data.ptr) for t in inputs],
        [list(t.shape) for t in inputs],
        int(out.data.ptr), list(out.shape),
        attrs, int(get_stream_ptr(stream))
    )
    return out
