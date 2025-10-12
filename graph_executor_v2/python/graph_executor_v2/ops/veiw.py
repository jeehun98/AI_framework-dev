# python/graph_executor_v2/ops/view.py
from __future__ import annotations
from typing import Tuple
import cupy as cp
from .common import ensure_cuda_dlls
ensure_cuda_dlls()
from graph_executor_v2.ops import _ops_view as _g

def reshape_alias(x: cp.ndarray, new_shape: Tuple[int,...]) -> cp.ndarray:
    if not isinstance(x, cp.ndarray) or x.dtype!=cp.float32 or not x.flags.c_contiguous:
        raise TypeError("x must be float32 C-contiguous")
    # cupy는 가능하면 view를 반환; 필요 시 copy될 수 있음. 우리는 alias만 허용하려면 체크:
    y = x.reshape(new_shape)  # cupy는 같은 포인터를 유지(연속 메모리) -> alias
    a = _g.ViewAttrs(); a.rank=len(new_shape)
    for i in range(4): a.shape[i]=int(new_shape[i]) if i<len(new_shape) else 1
    _g.alias_check(int(x.data.ptr), x.nbytes, list(x.shape), int(y.data.ptr), list(y.shape), a)
    return y
