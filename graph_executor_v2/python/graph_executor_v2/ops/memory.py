# python/graph_executor_v2/ops/memory.py
from __future__ import annotations
from typing import Optional
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_memory as _g
except Exception as e:
    raise ImportError(
        "[ops.memory] _ops_memory 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_memory)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


def _assert_contig_cuda(a: cp.ndarray, name: str):
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray expected")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name}: must be C-contiguous")
    if a.size <= 0:
        raise ValueError(f"{name}: empty array is not allowed")


def fill_f32(dst: cp.ndarray, value: float, *, stream: Optional[int] = None) -> None:
    """dst(float32, contiguous)에 value를 씁니다. CUDA Graph 캡처 안전."""
    _assert_contig_cuda(dst, "dst")
    if dst.dtype != cp.float32:
        raise TypeError("dst: float32 required")
    sptr = int(get_stream_ptr(stream))
    shape = [int(x) for x in dst.shape]
    _g.fill_f32(int(dst.data.ptr), shape, float(value), sptr)


def fill_i32(dst: cp.ndarray, value: int, *, stream: Optional[int] = None) -> None:
    """dst(int32, contiguous)에 value를 씁니다. CUDA Graph 캡처 안전."""
    _assert_contig_cuda(dst, "dst")
    if dst.dtype != cp.int32:
        raise TypeError("dst: int32 required")
    sptr = int(get_stream_ptr(stream))
    shape = [int(x) for x in dst.shape]
    _g.fill_i32(int(dst.data.ptr), shape, int(value), sptr)
