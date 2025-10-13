# python/graph_executor_v2/ops/rmsnorm.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_rmsnorm as _g
except Exception as e:
    raise ImportError(
        "[ops.rmsnorm] _ops_rmsnorm 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_rmsnorm)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e

def _assert_f32_2d(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray expected")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 required")
    if x.ndim != 2:
        raise ValueError(f"{name}: 2D [M,N] required")

def _mk_attrs(eps: float):
    a = _g.RMSNormAttrs()
    a.eps = float(eps)
    return a

def forward(
    X: cp.ndarray,
    *,
    gamma: Optional[cp.ndarray] = None,  # [N]
    beta: Optional[cp.ndarray]  = None,  # [N]
    eps: float = 1e-6,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    _assert_f32_2d(X, "X")
    M, N = map(int, X.shape)
    Xc = cp.ascontiguousarray(X)

    if gamma is not None:
        if not (isinstance(gamma, cp.ndarray) and gamma.dtype==cp.float32 and gamma.ndim==1 and gamma.size==N):
            raise TypeError("gamma must be float32 [N]")
        gamma = cp.ascontiguousarray(gamma)
    if beta is not None:
        if not (isinstance(beta, cp.ndarray) and beta.dtype==cp.float32 and beta.ndim==1 and beta.size==N):
            raise TypeError("beta must be float32 [N]")
        beta = cp.ascontiguousarray(beta)

    if out is None:
        out = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != X.shape:
            raise ValueError("out shape must equal X shape")
        out = cp.ascontiguousarray(out)

    attrs = _mk_attrs(eps)
    sptr = int(get_stream_ptr(stream))

    _g.forward(
        int(Xc.data.ptr), [M, N],
        None if gamma is None else int(gamma.data.ptr), [N] if gamma is not None else [],
        None if beta  is None else int(beta.data.ptr),  [N] if beta  is not None else [],
        int(out.data.ptr), [M, N],
        attrs, sptr
    )
    return out

def backward(
    X: cp.ndarray,
    dY: cp.ndarray,
    *,
    gamma: Optional[cp.ndarray] = None,  # [N]
    need_dgamma: bool = False,
    need_dbeta: bool = False,
    eps: float = 1e-6,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,         # dX
) -> tuple[cp.ndarray, Optional[cp.ndarray], Optional[cp.ndarray]]:
    _assert_f32_2d(X, "X")
    _assert_f32_2d(dY, "dY")
    if dY.shape != X.shape:
        raise ValueError("dY shape must match X")

    M, N = map(int, X.shape)
    Xc  = cp.ascontiguousarray(X)
    dYc = cp.ascontiguousarray(dY)

    g = None
    if gamma is not None:
        if not (isinstance(gamma, cp.ndarray) and gamma.dtype==cp.float32 and gamma.ndim==1 and gamma.size==N):
            raise TypeError("gamma must be float32 [N]")
        g = cp.ascontiguousarray(gamma)

    if out is None:
        dX = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out, "out(dX)")
        if out.shape != X.shape:
            raise ValueError("out(dX) shape must equal X")
        dX = cp.ascontiguousarray(out)

    dgamma = cp.empty((N,), dtype=cp.float32) if need_dgamma else None
    dbeta  = cp.empty((N,), dtype=cp.float32) if need_dbeta  else None

    attrs = _mk_attrs(eps)
    sptr = int(get_stream_ptr(stream))

    _g.backward(
        int(Xc.data.ptr), [M, N],
        None if g is None else int(g.data.ptr), [N] if g is not None else [],
        int(dYc.data.ptr), [M, N],
        int(dX.data.ptr),  [M, N],
        None if dgamma is None else int(dgamma.data.ptr), [N] if dgamma is not None else [],
        None if dbeta  is None else int(dbeta.data.ptr),  [N] if dbeta  is not None else [],
        attrs, sptr
    )
    return dX, dgamma, dbeta
