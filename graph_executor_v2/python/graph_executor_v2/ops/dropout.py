# python/graph_executor_v2/ops/dropout.py
from __future__ import annotations
from typing import Optional, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_dropout as _g
except Exception as e:
    raise ImportError("[ops.dropout] _ops_dropout 바인딩을 찾을 수 없습니다.") from e


def _check_f32(a: cp.ndarray, name: str):
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요")
    if a.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요 (got {a.dtype})")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name}: C-contiguous 필요")


def _check_mask(mask: cp.ndarray, ref_shape, name: str = "mask"):
    if not isinstance(mask, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요")
    if mask.dtype != cp.int32:
        raise TypeError(f"{name}: int32 필요 (got {mask.dtype})")
    if not mask.flags.c_contiguous:
        raise ValueError(f"{name}: C-contiguous 필요")
    if tuple(mask.shape) != tuple(ref_shape):
        raise ValueError(f"{name}: shape mismatch, expected {tuple(ref_shape)}, got {tuple(mask.shape)}")


# --------------------- 편의 API (eager) ---------------------
def forward(
    x: cp.ndarray,
    p: float = 0.1,
    *,
    train: bool = True,
    scale_in_train: bool = True,
    seed: int = 0x1234,
    counter_base: int = 0,
    out: Optional[cp.ndarray] = None,
    mask_out: Optional[cp.ndarray] = None,
    stream: Optional[int] = None,
) -> Dict[str, cp.ndarray]:
    """
    편의용 eager forward. out/mask_out 없으면 내부에서 생성.
    train=False면 p=0으로 동작.
    """
    _check_f32(x, "x")
    if out is None:
        out = cp.empty_like(x)
    else:
        _check_f32(out, "out")
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError("out shape mismatch")

    mask_ptr = None
    if mask_out is None:
        mask_out = cp.empty(x.shape, dtype=cp.int32)
    else:
        _check_mask(mask_out, x.shape, "mask_out")
        mask_ptr = int(mask_out.data.ptr)

    attrs = _g.DropoutAttrs()
    attrs.p = float(p if train else 0.0)
    attrs.seed = int(seed)
    attrs.scale_in_train = bool(scale_in_train)
    attrs.counter_base = int(counter_base)

    sptr = int(get_stream_ptr(stream))
    _g.forward(
        int(x.data.ptr), list(map(int, x.shape)),
        int(out.data.ptr), list(map(int, out.shape)),
        (mask_ptr if mask_ptr is not None else None),
        attrs, sptr
    )
    return {"y": out, "mask": mask_out}


def backward(
    dy: cp.ndarray,
    mask: cp.ndarray,
    *,
    p: float,
    scale_in_train: bool = True,
    stream: Optional[int] = None,
) -> cp.ndarray:
    _check_f32(dy, "dy")
    _check_mask(mask, dy.shape, "mask")
    dx = cp.empty_like(dy)

    attrs = _g.DropoutAttrs()
    attrs.p = float(p)
    attrs.scale_in_train = bool(scale_in_train)

    sptr = int(get_stream_ptr(stream))
    _g.backward(
        int(dy.data.ptr), list(map(int, dy.shape)),
        int(mask.data.ptr), list(map(int, mask.shape)),
        int(dx.data.ptr), list(map(int, dx.shape)),
        attrs, sptr
    )
    return dx


# --------------------- 캡처-세이프 API ---------------------
def forward_into(
    x: cp.ndarray,
    *,
    y: cp.ndarray,
    mask: Optional[cp.ndarray],
    p: float,
    scale_in_train: bool = True,
    seed: int = 0x1234,
    counter_base: int = 0,
    stream: Optional[int] = None,
) -> None:
    _check_f32(x, "x"); _check_f32(y, "y")
    if tuple(y.shape) != tuple(x.shape):
        raise ValueError("y/x shape mismatch")

    mask_ptr = None
    if mask is not None:
        _check_mask(mask, x.shape, "mask")
        mask_ptr = int(mask.data.ptr)

    attrs = _g.DropoutAttrs()
    attrs.p = float(p)
    attrs.scale_in_train = bool(scale_in_train)
    attrs.seed = int(seed)
    attrs.counter_base = int(counter_base)

    sptr = int(get_stream_ptr(stream))
    _g.forward(
        int(x.data.ptr), list(map(int, x.shape)),
        int(y.data.ptr), list(map(int, y.shape)),
        (mask_ptr if mask_ptr is not None else None),
        attrs, sptr
    )


def backward_into(
    dy: cp.ndarray,
    mask: cp.ndarray,
    *,
    dx: cp.ndarray,
    p: float,
    scale_in_train: bool = True,
    stream: Optional[int] = None,
) -> None:
    _check_f32(dy, "dy"); _check_f32(dx, "dx")
    if tuple(dx.shape) != tuple(dy.shape):
        raise ValueError("dx/dy shape mismatch")
    _check_mask(mask, dy.shape, "mask")

    attrs = _g.DropoutAttrs()
    attrs.p = float(p)
    attrs.scale_in_train = bool(scale_in_train)

    sptr = int(get_stream_ptr(stream))
    _g.backward(
        int(dy.data.ptr), list(map(int, dy.shape)),
        int(mask.data.ptr), list(map(int, mask.shape)),
        int(dx.data.ptr), list(map(int, dx.shape)),
        attrs, sptr
    )
