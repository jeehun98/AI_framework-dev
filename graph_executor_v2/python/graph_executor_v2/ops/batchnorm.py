# python/graph_executor_v2/ops/batchnorm.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    # C++ 바인딩 (_ops_batchnorm)
    from graph_executor_v2.ops import _ops_batchnorm as _g
except Exception as e:
    raise ImportError(
        "[ops.batchnorm] _ops_batchnorm 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_batchnorm)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# --------------------------- helpers ---------------------------
def _assert_f32_4d(x: cp.ndarray, name: str = "array"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"{name}: expected 4D, got shape={x.shape}")
    if not x.flags.c_contiguous:
        raise ValueError(f"{name}: must be C-contiguous")

def _assert_f32_1d(x: cp.ndarray, length: int, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 1 or int(x.size) != int(length):
        raise ValueError(f"{name}: expected 1D length={length}, got shape={x.shape}")
    if not x.flags.c_contiguous:
        raise ValueError(f"{name}: must be C-contiguous")

def _attrs_from_args(
    *,
    channels_last: bool,
    eps: float,
    momentum: float,
    training: bool,
    with_affine: bool,
) -> _g.BatchNormAttrs:
    a = _g.BatchNormAttrs()
    a.channels_last = bool(channels_last)
    a.eps           = float(eps)
    a.momentum      = float(momentum)
    a.training      = bool(training)
    a.with_affine   = bool(with_affine)
    # 고정값(필요 시 노출 가능)
    a.use_welford   = True
    a.num_groups    = 1
    return a


# --------------------------- public API (non-capture) ---------------------------
def forward(
    X: cp.ndarray,                         # [N,C,H,W] (channels_last=False) or [N,H,W,C] (True)
    running_mean: cp.ndarray,              # [C]
    running_var: cp.ndarray,               # [C]
    *,
    gamma: Optional[cp.ndarray] = None,    # [C] or None if with_affine=False
    beta:  Optional[cp.ndarray] = None,    # [C] or None if with_affine=False
    channels_last: bool = False,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
    with_affine: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> Tuple[cp.ndarray, Optional[cp.ndarray], cp.ndarray]:
    """
    BatchNorm forward.
    Returns: (Y, save_mean, save_invstd)
      - training=True  -> save_mean, save_invstd 둘 다 반환
      - training=False -> save_mean=None, save_invstd 반환(추론에서도 invstd 버퍼 필요)
    """
    _assert_f32_4d(X, "X")
    N, d1, d2, d3 = map(int, X.shape)
    if channels_last:
        C, H, W = d3, d1, d2
    else:
        C, H, W = d1, d2, d3

    _assert_f32_1d(running_mean, C, "running_mean")
    _assert_f32_1d(running_var,  C, "running_var")

    if with_affine:
        if gamma is None or beta is None:
            raise ValueError("with_affine=True requires gamma and beta")
        _assert_f32_1d(gamma, C, "gamma")
        _assert_f32_1d(beta,  C, "beta")
    else:
        if gamma is not None or beta is not None:
            raise ValueError("with_affine=False requires gamma=None and beta=None")

    if out is None:
        out = cp.empty_like(X)
    else:
        _assert_f32_4d(out, "out")
        if tuple(out.shape) != tuple(X.shape):
            raise ValueError(f"out shape must be {X.shape}")

    # save_* 준비 (바인딩 요구 사항에 맞춤)
    save_mean = cp.empty((C,), dtype=cp.float32) if training else None
    save_invstd = cp.empty((C,), dtype=cp.float32)

    attrs = _attrs_from_args(
        channels_last=channels_last, eps=eps, momentum=momentum,
        training=training, with_affine=with_affine
    )
    sptr = int(get_stream_ptr(stream))

    _g.forward(
        int(X.data.ptr), list(X.shape),
        int(out.data.ptr), list(out.shape),
        int(gamma.data.ptr) if gamma is not None else None,
        int(beta.data.ptr)  if beta  is not None else None,
        int(running_mean.data.ptr),
        int(running_var.data.ptr),
        attrs, sptr,
        int(save_mean.data.ptr)   if save_mean   is not None else None,
        int(save_invstd.data.ptr)
    )
    return out, save_mean, save_invstd


def backward(
    dY: cp.ndarray,                        # [N,C,H,W] or [N,H,W,C]
    X: cp.ndarray,                         # same layout as dY
    save_mean: cp.ndarray,                 # [C] (from forward training)
    save_invstd: cp.ndarray,               # [C] (from forward training or inference)
    *,
    gamma: Optional[cp.ndarray] = None,    # [C] if with_affine=True
    with_affine: bool = True,
    channels_last: bool = False,
    want_dX: bool = True,
    want_dgamma: bool = True,
    want_dbeta: bool = True,
    stream: Optional[int] = None,
) -> Dict[str, cp.ndarray]:
    """
    BatchNorm backward.
    Returns dict with keys subset of {"dX", "dgamma", "dbeta"}.
    """
    for name, t in (("dY", dY), ("X", X)):
        _assert_f32_4d(t, name)
    if tuple(dY.shape) != tuple(X.shape):
        raise ValueError("dY and X must have identical shapes")

    N, d1, d2, d3 = map(int, X.shape)
    if channels_last:
        C = d3
    else:
        C = d1

    _assert_f32_1d(save_mean,   C, "save_mean")
    _assert_f32_1d(save_invstd, C, "save_invstd")

    if with_affine:
        if gamma is None:
            raise ValueError("with_affine=True requires gamma")
        _assert_f32_1d(gamma, C, "gamma")
    else:
        if gamma is not None:
            raise ValueError("with_affine=False requires gamma=None")

    dX = cp.empty_like(X) if want_dX else None
    dgamma = cp.empty((C,), dtype=cp.float32) if want_dgamma else None
    dbeta  = cp.empty((C,), dtype=cp.float32) if want_dbeta  else None

    # attrs.training은 역전파 유무와 직접 연관되지 않지만,
    # 커널 구현 통일성을 위해 False/True 아무거나 가능. 여기서는 True로 둠.
    attrs = _attrs_from_args(
        channels_last=channels_last, eps=1e-5, momentum=0.1,
        training=True, with_affine=with_affine
    )
    sptr = int(get_stream_ptr(stream))

    _g.backward(
        int(dY.data.ptr), list(dY.shape),
        int(X.data.ptr),  list(X.shape),
        int(gamma.data.ptr) if (with_affine and gamma is not None) else None,
        int(save_mean.data.ptr),
        int(save_invstd.data.ptr),
        int(dX.data.ptr)      if dX      is not None else None,
        int(dgamma.data.ptr)  if dgamma  is not None else None,
        int(dbeta.data.ptr)   if dbeta   is not None else None,
        attrs, sptr
    )

    out: Dict[str, cp.ndarray] = {}
    if dX is not None: out["dX"] = dX
    if dgamma is not None: out["dgamma"] = dgamma
    if dbeta  is not None: out["dbeta"]  = dbeta
    return out
