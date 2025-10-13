# python/graph_executor_v2/ops/layernorm.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # C++ 바인딩 모듈
    from graph_executor_v2.ops import _ops_layernorm as _g
except Exception as e:
    raise ImportError(
        "[ops.layernorm] _ops_layernorm 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_layernorm)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# -------------------- helpers --------------------
def _assert_f32_2d(x: cp.ndarray, name: str) -> Tuple[int, int]:
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D (M,N), got shape={x.shape}")
    return int(x.shape[0]), int(x.shape[1])


def _assert_f32_1d(x: cp.ndarray, name: str) -> int:
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"{name}: expected 1D (N,), got shape={x.shape}")
    return int(x.shape[0])


def _mk_attrs(eps: float) -> "_g.LayerNormAttrs":
    a = _g.LayerNormAttrs()
    a.eps = float(eps)
    return a


def _pack_vec_opt(x: Optional[cp.ndarray], expected_N: int, name: str):
    """
    바인딩은 (ptr, shape)로 받지 않고 인자 2개(포인터-or-None, shape-list)를 따로 받음.
    - x가 None이면 (None, []) 반환.
    - 있으면 float32, 1D, 길이 N 체크 후 (ptr, [N]) 반환.
    """
    if x is None:
        return None, []  # (py::none, 빈 shape)
    N = _assert_f32_1d(x, name)
    if N != expected_N:
        raise ValueError(f"{name}: length {N} must equal feature N={expected_N}")
    x = cp.ascontiguousarray(x)
    return int(x.data.ptr), [N]


# -------------------- public api --------------------
def forward(
    X: cp.ndarray,                  # (M,N)
    *,
    gamma: Optional[cp.ndarray] = None,  # (N,) or None
    beta:  Optional[cp.ndarray] = None,  # (N,) or None
    eps: float = 1e-5,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    LayerNorm forward
      - X: float32 (M,N)
      - gamma/beta: 선택 (N,)
      - 반환: Y: (M,N)
    """
    M, N = _assert_f32_2d(X, "X")
    Xc = cp.ascontiguousarray(X)

    if out is None:
        out = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != X.shape:
            raise ValueError(f"out shape {out.shape} must equal X shape {X.shape}")
        out = cp.ascontiguousarray(out)

    g_ptr, g_shape = _pack_vec_opt(gamma, N, "gamma")  # (None, []) 허용
    b_ptr, b_shape = _pack_vec_opt(beta,  N, "beta")

    attrs = _mk_attrs(eps)
    sptr = int(get_stream_ptr(stream))

    _g.forward(
        int(Xc.data.ptr), [M, N],
        g_ptr, g_shape,
        b_ptr, b_shape,
        int(out.data.ptr), [M, N],
        attrs,
        sptr,
        None,  # ws_fwd (현재 미사용)
    )
    return out


def backward(
    X: cp.ndarray,          # (M,N)
    dY: cp.ndarray,         # (M,N)
    *,
    gamma: Optional[cp.ndarray] = None,    # (N,) or None
    eps: float = 1e-5,
    stream: Optional[int] = None,
    out_dx: Optional[cp.ndarray] = None,   # (M,N)
    out_dgamma: Optional[cp.ndarray] = None,  # (N,) or None → None이면 계산 안 함
    out_dbeta: Optional[cp.ndarray] = None,   # (N,) or None → None이면 계산 안 함
    return_param_grads: bool = False,      # True면 dgamma/dbeta 자동 할당
):
    """
    LayerNorm backward
      - 입력: X, dY (둘 다 float32, (M,N))
      - 선택: gamma (N,)
      - 출력:
          dX (M,N),
          dgamma (N,) | None,
          dbeta  (N,) | None
    """
    M, N = _assert_f32_2d(X, "X")
    M2, N2 = _assert_f32_2d(dY, "dY")
    if (M2, N2) != (M, N):
        raise ValueError("dY shape must match X shape")

    Xc  = cp.ascontiguousarray(X)
    dYc = cp.ascontiguousarray(dY)

    if out_dx is None:
        out_dx = cp.empty_like(Xc)
    else:
        _assert_f32_2d(out_dx, "out_dx")
        if out_dx.shape != X.shape:
            raise ValueError(f"out_dx shape {out_dx.shape} must equal X shape {X.shape}")
        out_dx = cp.ascontiguousarray(out_dx)

    g_ptr, g_shape = _pack_vec_opt(gamma, N, "gamma")

    # dgamma/dbeta: None이면 미계산. 반환 원하면 버퍼 제공 or 자동 할당
    dgamma_ptr, dgamma_shape = None, []
    dbeta_ptr,  dbeta_shape  = None, []

    dgamma_arr = None
    dbeta_arr  = None

    if out_dgamma is not None or return_param_grads:
        dgamma_arr = out_dgamma if out_dgamma is not None else cp.empty((N,), dtype=cp.float32)
        _assert_f32_1d(dgamma_arr, "out_dgamma")
        if dgamma_arr.size != N:
            raise ValueError("out_dgamma length must equal N")
        dgamma_arr = cp.ascontiguousarray(dgamma_arr)
        dgamma_ptr, dgamma_shape = int(dgamma_arr.data.ptr), [N]

    if out_dbeta is not None or return_param_grads:
        dbeta_arr = out_dbeta if out_dbeta is not None else cp.empty((N,), dtype=cp.float32)
        _assert_f32_1d(dbeta_arr, "out_dbeta")
        if dbeta_arr.size != N:
            raise ValueError("out_dbeta length must equal N")
        dbeta_arr = cp.ascontiguousarray(dbeta_arr)
        dbeta_ptr, dbeta_shape = int(dbeta_arr.data.ptr), [N]

    attrs = _mk_attrs(eps)
    sptr = int(get_stream_ptr(stream))

    _g.backward(
        int(Xc.data.ptr),  [M, N],
        g_ptr,             g_shape,
        int(dYc.data.ptr), [M, N],
        int(out_dx.data.ptr), [M, N],
        dgamma_ptr, dgamma_shape,
        dbeta_ptr,  dbeta_shape,
        attrs,
        sptr,
        None,  # ws_bwd (현재 미사용)
    )

    return out_dx, dgamma_arr, dbeta_arr
