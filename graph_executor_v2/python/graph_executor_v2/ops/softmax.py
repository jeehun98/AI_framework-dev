# python/graph_executor_v2/ops/softmax.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # 바인딩: SoftmaxAttrs(scale, log), softmax_forward, softmax_backward
    from graph_executor_v2.ops import _ops_softmax as _g
except Exception as e:
    raise ImportError(
        "[ops.softmax] _ops_softmax 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_softmax)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# -------------------- local helpers --------------------
def _assert_f32_2d(x: cp.ndarray, name: str) -> Tuple[int, int]:
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D (M,N), got shape={x.shape}")
    return int(x.shape[0]), int(x.shape[1])


def _pack_mask(mask: Optional[cp.ndarray]) -> Optional[tuple[int, list[int]]]:
    """
    바인딩이 요구하는 mask 인자 형태: (uintptr_t, [M,N] or [1,N] or [M,1])
    - None 이면 None 전달
    - dtype=float32, ndim=2만 허용
    """
    if mask is None:
        return None
    if not isinstance(mask, cp.ndarray) or mask.dtype != cp.float32 or mask.ndim != 2:
        raise TypeError("mask must be a CuPy float32 2D array or None")
    return (int(mask.data.ptr), [int(d) for d in mask.shape])


def _attrs(scale: float, log: bool):
    a = _g.SoftmaxAttrs()
    a.scale = float(scale)
    a.log = bool(log)
    return a


# -------------------- public api --------------------
def forward(
    X: cp.ndarray,                  # (M,N) logits or scores
    *,
    mask: Optional[cp.ndarray] = None,  # (M,N) or (1,N) or (M,1), float32
    scale: float = 1.0,
    log: bool = False,              # True면 LogSoftmax
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Softmax / LogSoftmax (forward)
      - 입력/출력: float32 (M,N)
      - mask(선택): (M,N) 또는 브로드캐스트 호환 (1,N)/(M,1), float32
      - scale: 입력 로그릿에 곱할 스케일(예: 1/sqrt(d_k) 같은 SDPA 스케일)
      - log=True 이면 LogSoftmax
    """
    M, N = _assert_f32_2d(X, "X")
    if out is None:
        out = cp.empty_like(X)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != X.shape:
            raise ValueError(f"out shape {out.shape} must equal X shape {X.shape}")

    mask_arg = _pack_mask(mask)
    attrs = _attrs(scale, log)
    sptr = int(get_stream_ptr(stream))

    _g.softmax_forward(
        int(X.data.ptr), [M, N],
        int(out.data.ptr), [M, N],
        mask_arg, attrs, sptr
    )
    return out


def logsoftmax(
    X: cp.ndarray,
    *,
    mask: Optional[cp.ndarray] = None,
    scale: float = 1.0,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """편의 함수: log=True로 forward 호출."""
    return forward(X, mask=mask, scale=scale, log=True, stream=stream, out=out)


def backward(
    Y_or_X: cp.ndarray,             # (M,N) : y_provided=True -> Y, False -> X
    dY: cp.ndarray,                 # (M,N)
    *,
    mask: Optional[cp.ndarray] = None,
    scale: float = 1.0,
    log: bool = False,
    y_provided: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,   # dX (M,N) 버퍼
) -> cp.ndarray:
    """
    Softmax / LogSoftmax (backward)
      - y_provided=True  : 첫 인자는 forward 출력 Y
      - y_provided=False : 첫 인자는 forward 입력 X (내부에서 forward 재계산)
      - 반환: dX (M,N)
    """
    M, N = _assert_f32_2d(Y_or_X, "Y_or_X")
    M2, N2 = _assert_f32_2d(dY, "dY")
    if (M2, N2) != (M, N):
        raise ValueError("dY shape must match Y_or_X shape")

    if out is None:
        out = cp.empty_like(Y_or_X)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != Y_or_X.shape:
            raise ValueError(f"out shape {out.shape} must equal Y_or_X shape {Y_or_X.shape}")

    mask_arg = _pack_mask(mask)
    attrs = _attrs(scale, log)
    sptr = int(get_stream_ptr(stream))

    _g.softmax_backward(
        int(Y_or_X.data.ptr), [M, N],
        int(dY.data.ptr),     [M, N],
        int(out.data.ptr),    [M, N],
        mask_arg, attrs, bool(y_provided), sptr
    )
    return out
