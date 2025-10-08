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


def _ensure_contig(a: cp.ndarray, name: str) -> cp.ndarray:
    """C-연속이 아니면 연속 버퍼로 변환(캡처-세이프·커널 가정 일치)."""
    if not a.flags.c_contiguous:
        a = cp.ascontiguousarray(a)
        if a.dtype != cp.float32:
            a = a.astype(cp.float32, copy=False)
    return a


def _pack_mask(mask: Optional[cp.ndarray]) -> Optional[tuple[int, list[int]]]:
    """
    바인딩이 요구하는 mask 인자 형태: (uintptr_t, shape)
      허용 shape: [M,N], [1,N], [M,1], [N]
      dtype=float32, ndim in {1,2}
    """
    if mask is None:
        return None
    if not isinstance(mask, cp.ndarray) or mask.dtype != cp.float32:
        raise TypeError("mask must be a CuPy float32 array or None")
    if mask.ndim not in (1, 2):
        raise TypeError("mask.ndim must be 1 or 2 (allowed: [N], [M,N], [1,N], [M,1])")
    mask = _ensure_contig(mask, "mask")
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
    mask: Optional[cp.ndarray] = None,  # [M,N] / [1,N] / [M,1] / [N], float32
    scale: float = 1.0,
    log: bool = False,              # True면 LogSoftmax
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Softmax / LogSoftmax (forward)
      - 입력/출력: float32 (M,N), C-연속 필요(비연속이면 내부에서 연속화)
      - mask(선택): [M,N] / [1,N] / [M,1] / [N], float32
      - scale: 입력에 곱할 스케일(예: 1/sqrt(d_k))
      - log=True 이면 LogSoftmax
    """
    M, N = _assert_f32_2d(X, "X")

    Xc = _ensure_contig(X, "X")

    if out is None:
        out = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != X.shape:
            raise ValueError(f"out shape {out.shape} must equal X shape {X.shape}")
        out = _ensure_contig(out, "out")

    mask_arg = _pack_mask(mask)
    attrs = _attrs(scale, log)
    sptr = int(get_stream_ptr(stream))

    _g.softmax_forward(
        int(Xc.data.ptr), [M, N],
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
    Y_or_X: cp.ndarray,             # (M,N) : 현재 커널은 Y만 지원(y_provided=True)
    dY: cp.ndarray,                 # (M,N)
    *,
    mask: Optional[cp.ndarray] = None,  # 현재 커널에서 무시되지만 형태 검증 용도로 허용
    scale: float = 1.0,
    log: bool = False,
    y_provided: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,   # dX (M,N) 버퍼
) -> cp.ndarray:
    """
    Softmax / LogSoftmax (backward)
      - 현재 커널은 y_provided=True만 지원(Y=forward 출력 필요)
      - 반환: dX (M,N)
    """
    if not y_provided:
        raise ValueError("softmax.backward: y_provided=False (재계산 경로)는 현재 미지원입니다. "
                         "forward 출력 Y를 전달하고 y_provided=True로 호출하세요.")

    M, N = _assert_f32_2d(Y_or_X, "Y_or_X")
    M2, N2 = _assert_f32_2d(dY, "dY")
    if (M2, N2) != (M, N):
        raise ValueError("dY shape must match Y_or_X shape")

    Yc = _ensure_contig(Y_or_X, "Y_or_X")
    dYc = _ensure_contig(dY, "dY")

    if out is None:
        out = cp.empty_like(Yc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != Y_or_X.shape:
            raise ValueError(f"out shape {out.shape} must equal Y_or_X shape {Y_or_X.shape}")
        out = _ensure_contig(out, "out")

    # mask는 bwd에서 무시되지만, 바인딩은 검증을 수행하므로 포맷만 맞춰 전달
    mask_arg = _pack_mask(mask)
    attrs = _attrs(scale, log)
    sptr = int(get_stream_ptr(stream))

    _g.softmax_backward(
        int(Yc.data.ptr), [M, N],
        int(dYc.data.ptr), [M, N],
        int(out.data.ptr), [M, N],
        mask_arg, attrs, True, sptr
    )
    return out
