from __future__ import annotations
from typing import Optional, Sequence, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_common as _c
    from graph_executor_v2.ops import _ops_slice as _g
except Exception as e:
    raise ImportError("[ops.slice] _ops_slice 바인딩을 찾을 수 없습니다.") from e


# --------------------------- helpers ---------------------------
def _check_f32_4d(x: cp.ndarray, name="x"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요 (got {x.dtype})")
    if x.ndim != 4:
        raise ValueError(f"{name}: 4D 필요 (N,C,H,W). got {x.ndim}D")


def _canonize(start: Sequence[int], end: Sequence[int], step: Sequence[int], shape: Tuple[int, int, int, int]):
    if len(start) != 4 or len(end) != 4 or len(step) != 4:
        raise ValueError("start/end/step must be len-4 sequences")
    N, C, H, W = map(int, shape)
    S = list(map(int, start))
    E = list(map(int, end))
    T = list(map(int, step))
    for i, dim in enumerate((N, C, H, W)):
        if T[i] == 0:
            raise ValueError("step cannot be zero")
        # clamp-ish (런처쪽에서도 clamp 옵션 있음)
        if T[i] > 0:
            S[i] = max(0, min(S[i], dim))
            E[i] = max(0, min(E[i], dim))
        else:
            S[i] = max(-dim, min(S[i], dim - 1))
            E[i] = max(-dim - 1, min(E[i], dim - 1))
    return S, E, T


def _infer_out_shape(start, end, step, in_shape):
    # PyTorch-style 배제: 여기선 단순 등차 슬라이스 크기 계산
    out = []
    for s, e, t, dim in zip(start, end, step, in_shape):
        if t > 0:
            length = max(0, (e - s + (t - 1)) // t)
        else:
            length = max(0, (s - e - 1) // (-t) + 1)
        out.append(length)
    return tuple(out)


# --------------------------- API ---------------------------
def forward(x: cp.ndarray,
            start: Sequence[int], end: Sequence[int], step: Sequence[int],
            out: Optional[cp.ndarray] = None,
            clamp: bool = True,
            stream: Optional[int] = None) -> cp.ndarray:
    """
    4D Slice (float32), inclusive-exclusive 범위.
    """
    _check_f32_4d(x, "x")
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    S, E, T = _canonize(start, end, step, x.shape)
    out_shape = _infer_out_shape(S, E, T, x.shape)

    if out is None:
        out = cp.empty(out_shape, dtype=cp.float32)
    else:
        if out.dtype != cp.float32 or tuple(out.shape) != out_shape or not out.flags.c_contiguous:
            raise ValueError("out shape/dtype/contiguity mismatch")

    attrs = _g.SliceAttrs()
    attrs.clamp = bool(clamp)
    # 바인딩에서 배열은 set_* 함수로 복사
    attrs.set_start(list(map(int, S)))
    attrs.set_end(list(map(int, E)))
    attrs.set_step(list(map(int, T)))

    sptr = int(get_stream_ptr(stream))
    _g.forward(int(x.data.ptr), list(map(int, x.shape)),
               int(out.data.ptr), list(map(int, out_shape)),
               attrs, sptr)
    return out


def backward(x: cp.ndarray,
             start: Sequence[int], end: Sequence[int], step: Sequence[int],
             gy: cp.ndarray,
             stream: Optional[int] = None) -> cp.ndarray:
    """
    Slice backward: scatter-add gy into gx (float32).
    """
    _check_f32_4d(x, "x")
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    if not isinstance(gy, cp.ndarray) or gy.dtype != cp.float32 or gy.ndim != 4:
        raise ValueError("gy must be 4D float32")
    if not gy.flags.c_contiguous:
        gy = cp.ascontiguousarray(gy)

    S, E, T = _canonize(start, end, step, x.shape)
    expected = _infer_out_shape(S, E, T, x.shape)
    if tuple(gy.shape) != expected:
        raise ValueError(f"gy shape mismatch: expected {expected}, got {gy.shape}")

    gx = cp.zeros_like(x)

    attrs = _g.SliceAttrs()
    attrs.clamp = True
    attrs.set_start(list(map(int, S)))
    attrs.set_end(list(map(int, E)))
    attrs.set_step(list(map(int, T)))

    sptr = int(get_stream_ptr(stream))
    _g.backward(int(x.data.ptr), list(map(int, x.shape)),
                int(gy.data.ptr), list(map(int, gy.shape)),
                int(gx.data.ptr), attrs, sptr)
    return gx
