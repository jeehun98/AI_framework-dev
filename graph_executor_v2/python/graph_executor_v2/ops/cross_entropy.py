# python/graph_executor_v2/ops/cross_entropy.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # 바인딩: Reduction enum, CrossEntropyAttrs, forward(), backward()
    from graph_executor_v2.ops import _ops_cross_entropy as _g
except Exception as e:
    raise ImportError(
        "[ops.cross_entropy] _ops_cross_entropy 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_cross_entropy)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# -------------------- local helpers --------------------
def _assert_logits(x: cp.ndarray, name: str = "X") -> Tuple[int, int]:
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D (M,N), got shape={x.shape}")
    return int(x.shape[0]), int(x.shape[1])


def _as_int32_targets(t: cp.ndarray, name: str = "targets") -> cp.ndarray:
    if not isinstance(t, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(t)}")
    if t.ndim != 1:
        raise ValueError(f"{name}: expected 1D (M,), got shape={t.shape}")
    # 바인딩이 I32 고정 → 안전 캐스팅(복사없이 가능하면 no-copy)
    if t.dtype != cp.int32:
        t = t.astype(cp.int32, copy=False)
    return t


def _mk_attrs(
    *,
    from_logits: bool,
    reduction: str,
    ignore_index: int,
    eps: float,
    ls_eps: float,
):
    # reduction 문자열 → enum
    red = reduction.lower()
    if red == "none":
        red_enum = _g.Reduction.None_
    elif red == "mean":
        red_enum = _g.Reduction.Mean
    elif red == "sum":
        red_enum = _g.Reduction.Sum
    else:
        raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")

    a = _g.CrossEntropyAttrs()
    a.from_logits = bool(from_logits)
    a.reduction = red_enum
    a.ignore_index = int(ignore_index)
    a.eps = float(eps)
    a.ls_eps = float(ls_eps)
    return a


# -------------------- public api --------------------
def forward(
    X: cp.ndarray,           # (M,N), logits 또는 확률( from_logits=False 시 )
    targets: cp.ndarray,     # (M,) int32 (자동 int32 캐스팅)
    *,
    from_logits: bool = True,
    reduction: str = "mean",         # 'none' | 'mean' | 'sum'
    ignore_index: int = -1,
    eps: float = 1e-7,               # 확률 안정성용 epsilon
    ls_eps: float = 0.0,             # label smoothing epsilon
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None # reduction='none' -> (M,), else -> (1,)
) -> cp.ndarray:
    """
    Cross-Entropy forward
      - X: float32 (M,N)
      - targets: int32 (M,)
      - from_logits: True면 X를 logits로 간주(내부에서 softmax), False면 X를 확률로 간주
      - reduction: 'none' → (M,), 'mean'/'sum' → (1,)
    """
    M, N = _assert_logits(X, "X")
    T = _as_int32_targets(targets, "targets")
    if T.shape[0] != M:
        raise ValueError(f"targets length {T.shape[0]} must equal batch M={M}")

    attrs = _mk_attrs(
        from_logits=from_logits, reduction=reduction,
        ignore_index=ignore_index, eps=eps, ls_eps=ls_eps
    )
    sptr = int(get_stream_ptr(stream))

    # loss shape 결정
    if attrs.reduction == _g.Reduction.None_:
        expected = (M,)
    else:
        expected = (1,)

    if out is None:
        out = cp.empty(expected, dtype=cp.float32)
    else:
        if out.dtype != cp.float32:
            raise TypeError("out must be float32")
        if out.shape != expected:
            raise ValueError(f"out shape {out.shape} must be {expected}")

    _g.forward(
        int(X.data.ptr), [M, N],
        int(T.data.ptr), [M],
        int(out.data.ptr), [out.shape[0]],
        attrs, sptr
    )
    return out


def backward(
    X: cp.ndarray,           # (M,N)  (from_logits에 따라 의미가 달라짐)
    targets: cp.ndarray,     # (M,) int32
    *,
    from_logits: bool = True,
    reduction: str = "mean",   # backward는 dX만 반환하므로 reduction은 scale에 영향
    ignore_index: int = -1,
    eps: float = 1e-7,
    ls_eps: float = 0.0,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,  # dX (M,N)
) -> cp.ndarray:
    """
    Cross-Entropy backward
      - 반환: dX (M,N)
      - from_logits=True: d(logits), False: d(probabilities)
    """
    M, N = _assert_logits(X, "X")
    T = _as_int32_targets(targets, "targets")
    if T.shape[0] != M:
        raise ValueError(f"targets length {T.shape[0]} must equal batch M={M}")

    attrs = _mk_attrs(
        from_logits=from_logits, reduction=reduction,
        ignore_index=ignore_index, eps=eps, ls_eps=ls_eps
    )
    sptr = int(get_stream_ptr(stream))

    if out is None:
        out = cp.empty_like(X)
    else:
        _assert_logits(out, "out")
        if out.shape != X.shape:
            raise ValueError(f"out shape {out.shape} must equal X shape {X.shape}")

    _g.backward(
        int(X.data.ptr), [M, N],
        int(T.data.ptr), [M],
        int(out.data.ptr), [M, N],
        attrs, sptr
    )
    return out
