# python/graph_executor_v2/ops/cross_entropy.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
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


def _as_contig_f32_2d(x: cp.ndarray, name: str) -> cp.ndarray:
    # 바인딩은 항상 연속 row-major stride를 가정하므로 여기서 보장
    if x.dtype != cp.float32 or x.ndim != 2:
        raise TypeError(f"{name}: must be float32 2D")
    return cp.ascontiguousarray(x) if not x.flags.c_contiguous else x


def _as_int32_targets(t: cp.ndarray, name: str = "targets") -> cp.ndarray:
    if not isinstance(t, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(t)}")
    if t.ndim != 1:
        raise ValueError(f"{name}: expected 1D (M,), got shape={t.shape}")
    if t.dtype != cp.int32:
        t = t.astype(cp.int32, copy=False)
    # 연속 보장
    return cp.ascontiguousarray(t) if not t.flags.c_contiguous else t


def _mk_attrs(
    *,
    from_logits: bool,
    reduction: str,
    ignore_index: int,
    eps: float,
    ls_eps: float,
):
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
    X: cp.ndarray,           # (M,N), logits 또는 확률(from_logits=False)
    targets: cp.ndarray,     # (M,) int32
    *,
    from_logits: bool = True,
    reduction: str = "mean",         # 'none' | 'mean' | 'sum'
    ignore_index: int = -1,
    eps: float = 1e-7,               # 확률 안정성용 epsilon (from_logits=False에서 사용)
    ls_eps: float = 0.0,             # label smoothing epsilon
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None # reduction='none' -> (M,), else -> (1,)
) -> cp.ndarray:
    """
    Cross-Entropy forward (capture-safe)
      - X: float32 (M,N); 바인딩이 연속 stride를 가정하므로 내부에서 연속화
      - targets: int32 (M,); 내부에서 int32/연속 보장
      - from_logits: True면 내부 softmax; False면 X를 확률로 처리(eps로 안정화)
      - reduction: 'none' → (M,), 'mean'/'sum' → (1,)
      - ignore_index: 해당 라벨은 손실 0 (Mean 분모는 M으로 해석)
    """
    M, N = _assert_logits(X, "X")
    Xc = _as_contig_f32_2d(X, "X")
    T = _as_int32_targets(targets, "targets")
    if T.shape[0] != M:
        raise ValueError(f"targets length {T.shape[0]} must equal batch M={M}")

    attrs = _mk_attrs(
        from_logits=from_logits, reduction=reduction,
        ignore_index=ignore_index, eps=eps, ls_eps=ls_eps
    )
    sptr = int(get_stream_ptr(stream))

    expected = (M,) if attrs.reduction == _g.Reduction.None_ else (1,)
    if out is None:
        out = cp.empty(expected, dtype=cp.float32)
    else:
        if out.dtype != cp.float32:
            raise TypeError("out must be float32")
        if out.shape != expected:
            raise ValueError(f"out shape {out.shape} must be {expected}")
        if not out.flags.c_contiguous:
            out = cp.ascontiguousarray(out)

    _g.forward(
        int(Xc.data.ptr), [M, N],
        int(T.data.ptr),  [M],
        int(out.data.ptr), [out.shape[0]],
        attrs, sptr
    )
    return out


def backward(
    X: cp.ndarray,           # (M,N)  (from_logits에 따라 의미가 달라짐)
    targets: cp.ndarray,     # (M,) int32
    *,
    from_logits: bool = True,
    reduction: str = "mean",   # backward는 dX만 반환하므로 scale에만 영향
    ignore_index: int = -1,
    eps: float = 1e-7,
    ls_eps: float = 0.0,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,  # dX (M,N)
) -> cp.ndarray:
    """
    Cross-Entropy backward (capture-safe)
      - 반환: dX (M,N)
      - from_logits=True: d(logits), False: d(probabilities)
      - reduction='mean'이면 자동 1/M 스케일 적용
    """
    M, N = _assert_logits(X, "X")
    Xc = _as_contig_f32_2d(X, "X")
    T = _as_int32_targets(targets, "targets")
    if T.shape[0] != M:
        raise ValueError(f"targets length {T.shape[0]} must equal batch M={M}")

    attrs = _mk_attrs(
        from_logits=from_logits, reduction=reduction,
        ignore_index=ignore_index, eps=eps, ls_eps=ls_eps
    )
    sptr = int(get_stream_ptr(stream))

    if out is None:
        out = cp.empty_like(Xc)
    else:
        _assert_logits(out, "out")
        if out.shape != Xc.shape:
            raise ValueError(f"out shape {out.shape} must equal X shape {Xc.shape}")
        if not out.flags.c_contiguous:
            out = cp.ascontiguousarray(out)

    _g.backward(
        int(Xc.data.ptr), [M, N],
        int(T.data.ptr),  [M],
        int(out.data.ptr), [M, N],
        attrs, sptr
    )
    return out
