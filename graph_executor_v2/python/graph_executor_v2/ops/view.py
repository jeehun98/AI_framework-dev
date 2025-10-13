from __future__ import annotations
from typing import Optional, Sequence
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_common as _c
    from graph_executor_v2.ops import _ops_view as _g
except Exception as e:
    raise ImportError("[ops.view] _ops_view 바인딩을 찾을 수 없습니다.") from e


# --------------------------- helpers ---------------------------
def _check_f32(x: cp.ndarray, name="x"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요 (got {x.dtype})")


def _to_i32_list(seq: Sequence[int], n: int, name: str) -> list[int]:
    if len(seq) != n:
        raise ValueError(f"{name} must be len-{n}")
    v = [int(t) for t in seq]
    return v


# --------------------------- API ---------------------------
def forward(x: cp.ndarray,
            shape: Sequence[int],
            stride: Optional[Sequence[int]] = None,
            offset: int = 0,
            out: Optional[cp.ndarray] = None,
            stream: Optional[int] = None) -> cp.ndarray:
    """
    View: pointer reinterpret (no copy). shape/stride/offset 기준으로 출력 버퍼에 써 줌.
    - 실제론 device 커널을 통해 검증 및 copy(필요 시) 수행하는 안전한 경로를 사용.
    """
    _check_f32(x, "x")
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    shp = _to_i32_list(shape, len(shape), "shape")
    use_stride = stride is not None
    if use_stride:
        std = _to_i32_list(stride, len(shape), "stride")
    else:
        # contiguous stride
        std = [0] * len(shp)
        std[-1] = 1
        for i in range(len(shp) - 2, -1, -1):
            std[i] = std[i + 1] * shp[i + 1]

    if out is None:
        out = cp.empty(tuple(shp), dtype=cp.float32)
    else:
        _check_f32(out, "out")
        if tuple(out.shape) != tuple(shp):
            raise ValueError(f"out shape mismatch: expected {tuple(shp)}, got {out.shape}")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

    attrs = _g.ViewAttrs()
    attrs.set_shape(shp)
    attrs.set_stride(std)
    attrs.offset = int(offset)

    sptr = int(get_stream_ptr(stream))
    _g.forward(int(x.data.ptr), list(map(int, x.shape)),
               int(out.data.ptr), list(map(int, out.shape)),
               attrs, sptr)
    return out


def backward(x: cp.ndarray,
             shape: Sequence[int],
             stride: Optional[Sequence[int]],
             offset: int,
             gy: cp.ndarray,
             stream: Optional[int] = None) -> cp.ndarray:
    """
    View backward: gy -> gx gather-add (float32).
    """
    _check_f32(x, "x")
    _check_f32(gy, "gy")
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    if not gy.flags.c_contiguous:
        gy = cp.ascontiguousarray(gy)

    shp = _to_i32_list(shape, len(shape), "shape")
    if tuple(gy.shape) != tuple(shp):
        raise ValueError(f"gy shape mismatch: expected {tuple(shp)}, got {gy.shape}")

    if stride is not None:
        std = _to_i32_list(stride, len(shape), "stride")
    else:
        std = [0] * len(shp)
        std[-1] = 1
        for i in range(len(shp) - 2, -1, -1):
            std[i] = std[i + 1] * shp[i + 1]

    attrs = _g.ViewAttrs()
    attrs.set_shape(shp)
    attrs.set_stride(std)
    attrs.offset = int(offset)

    gx = cp.zeros_like(x)
    sptr = int(get_stream_ptr(stream))
    _g.backward(int(x.data.ptr), list(map(int, x.shape)),
                int(gy.data.ptr), list(map(int, gy.shape)),
                int(gx.data.ptr), attrs, sptr)
    return gx
