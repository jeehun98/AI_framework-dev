# python/graph_executor_v2/ops/optimizer.py
from __future__ import annotations
from typing import Optional
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_optimizer as _g
except Exception as e:
    raise ImportError(
        "[ops.optimizer] _ops_optimizer 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_optimizer)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# ------------------------- helpers -------------------------
def _assert_f32_1d(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray expected")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: float32 required")
    if x.ndim != 1:
        raise ValueError(f"{name}: 1D [N] required")
    if x.size <= 0:
        raise ValueError(f"{name}: length must be > 0")


def _mk_sgd_attrs(lr: float, momentum: float, dampening: float,
                  nesterov: bool, weight_decay: float):
    a = _g.SGDAttrs()
    a.lr = float(lr)
    a.momentum = float(momentum)
    a.dampening = float(dampening)
    a.nesterov = bool(nesterov)
    a.weight_decay = float(weight_decay)
    return a


def _mk_adamw_attrs(lr: float, beta1: float, beta2: float, eps: float,
                    weight_decay: float, bias_correction: bool, step: int):
    a = _g.AdamWAttrs()
    a.lr = float(lr)
    a.beta1 = float(beta1)
    a.beta2 = float(beta2)
    a.eps = float(eps)
    a.weight_decay = float(weight_decay)
    a.bias_correction = bool(bias_correction)
    a.step = int(step)
    return a


# ------------------------- public API -------------------------
def sgd_update(
    P: cp.ndarray,              # [N] params (in-place)
    G: cp.ndarray,              # [N] grads
    V: Optional[cp.ndarray] = None,  # [N] velocity (momentum>0이면 필요)
    *,
    lr: float = 1e-2,
    momentum: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    stream: Optional[int] = None,
) -> cp.ndarray:
    """
    In-place SGD(+Momentum/Nesterov, L2) 업데이트.
    반환값으로 편의상 P를 그대로 돌려준다.
    """
    _assert_f32_1d(P, "P")
    _assert_f32_1d(G, "G")
    if P.size != G.size:
        raise ValueError("P and G length must match")
    Pc = cp.ascontiguousarray(P)
    Gc = cp.ascontiguousarray(G)

    Vc = None
    if momentum > 0.0:
        if V is None:
            raise ValueError("momentum > 0.0 인 경우 V[velocity]가 필요합니다")
        _assert_f32_1d(V, "V")
        if V.size != P.size:
            raise ValueError("V length must match P")
        Vc = cp.ascontiguousarray(V)
    else:
        if V is not None:
            # 모멘텀 0인데 V가 들어온 경우: 무시하지만 contiguous 보장 불필요
            Vc = None

    attrs = _mk_sgd_attrs(lr, momentum, dampening, nesterov, weight_decay)
    sptr = int(get_stream_ptr(stream))

    _g.sgd_update(
        int(Pc.data.ptr), [int(Pc.size)],
        int(Gc.data.ptr), [int(Gc.size)],
        None if Vc is None else int(Vc.data.ptr),
        [] if Vc is None else [int(Vc.size)],
        attrs, sptr
    )
    # in-place 적용 → 원본 뷰가 Pc와 공유되므로 P도 업데이트됨
    return P


def adamw_update(
    P: cp.ndarray,  # [N] params (in-place)
    G: cp.ndarray,  # [N] grads
    M: cp.ndarray,  # [N] first moment (in-place)
    V: cp.ndarray,  # [N] second moment (in-place)
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
    step: int = 1,
    stream: Optional[int] = None,
) -> cp.ndarray:
    """
    In-place AdamW 업데이트. (Decoupled weight decay)
    반환값으로 편의상 P를 그대로 돌려준다.
    """
    _assert_f32_1d(P, "P")
    _assert_f32_1d(G, "G")
    _assert_f32_1d(M, "M")
    _assert_f32_1d(V, "V")
    N = P.size
    if G.size != N or M.size != N or V.size != N:
        raise ValueError("Lengths of P, G, M, V must all match")

    Pc = cp.ascontiguousarray(P)
    Gc = cp.ascontiguousarray(G)
    Mc = cp.ascontiguousarray(M)
    Vc = cp.ascontiguousarray(V)

    attrs = _mk_adamw_attrs(lr, beta1, beta2, eps, weight_decay, bias_correction, step)
    sptr = int(get_stream_ptr(stream))

    _g.adamw_update(
        int(Pc.data.ptr), [int(N)],
        int(Gc.data.ptr), [int(N)],
        int(Mc.data.ptr), [int(N)],
        int(Vc.data.ptr), [int(N)],
        attrs, sptr
    )
    return P
