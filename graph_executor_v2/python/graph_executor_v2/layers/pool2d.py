# python/graph_executor_v2/ops/pool2d.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp
from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

# --- 바인딩 모듈 로드(여기까지는 OK) ---
try:
    from graph_executor_v2.ops import _ops_pool2d as _g
except Exception as e:
    raise ImportError(
        "[ops.pool2d] _ops_pool2d 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_pool2d)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e

# ---- 즉시 탐지/초기화 금지! 필요한 시점에만 가져오기 ----
def _get_make_tensor_4d():
    fn = getattr(_g, "make_tensor_4d", None)
    if fn is not None:
        return fn
    try:
        from graph_executor_v2.ops import _ops_common as _c
        fn = getattr(_c, "make_tensor_4d", None)
    except Exception:
        fn = None
    if fn is None:
        raise RuntimeError(
            "[ops.pool2d] make_tensor_4d 팩토리를 찾지 못했습니다. "
            "_ops_pool2d 또는 _ops_common 중 하나에 make_tensor_4d가 필요합니다."
        )
    return fn

def _assert_f32_4d(x: cp.ndarray, name: str = "array"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"{name}: expected 4D (N,C,H,W), got shape={x.shape}")

def _as_tensor_4d(x: cp.ndarray):
    _assert_f32_4d(x, "as_tensor_4d(x)")
    N, C, H, W = map(int, x.shape)
    sN = int(x.strides[0] // x.itemsize)
    sC = int(x.strides[1] // x.itemsize)
    sH = int(x.strides[2] // x.itemsize)
    sW = int(x.strides[3] // x.itemsize)
    MAKE_T4D = _get_make_tensor_4d()   # ← 함수 내부에서 지연 획득
    return MAKE_T4D(int(x.data.ptr), N, C, H, W, sN, sC, sH, sW)

# --- fwd/bwd 엔트리도 호출 시점에 확인 ---
def _has(name: str) -> bool:
    return hasattr(_g, name)

def forward(
    X: cp.ndarray,
    *,
    kernel: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (2, 2),
    padding: Tuple[int, int] = (0, 0),
    mode: str = "max",
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    _assert_f32_4d(X, "X")
    N, C, H, W = map(int, X.shape)
    kH, kW = map(int, kernel)
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    m = mode.lower()
    if m not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    mode_id = 0 if m == "max" else 1

    outH = (H + 2*pH - kH)//sH + 1
    outW = (W + 2*pW - kW)//sW + 1
    if out is None:
        out = cp.empty((N, C, outH, outW), dtype=cp.float32)

    tX = _as_tensor_4d(X)
    tY = _as_tensor_4d(out)
    sptr = int(get_stream_ptr(stream))

    if _has("forward_ex"):
        _g.forward_ex(tX, tY, kH, kW, sH, sW, pH, pW, mode_id, sptr)
        return out
    if _has("forward"):
        _g.forward(tX, tY, kH, kW, sH, sW, pH, pW, mode_id, sptr)
        return out
    raise NotImplementedError("[ops.pool2d] 바인딩에 forward_ex/forward 함수가 없습니다.")

def backward(
    X: cp.ndarray,
    Y: cp.ndarray,
    gY: cp.ndarray,
    *,
    kernel: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (2, 2),
    padding: Tuple[int, int] = (0, 0),
    mode: str = "max",
    stream: Optional[int] = None,
) -> cp.ndarray:
    _assert_f32_4d(X, "X")
    _assert_f32_4d(Y, "Y")
    _assert_f32_4d(gY, "gY")

    dx = cp.empty_like(X)
    tX  = _as_tensor_4d(X)
    tY  = _as_tensor_4d(Y)
    tgY = _as_tensor_4d(gY)
    t_dX = _as_tensor_4d(dx)

    kH, kW = map(int, kernel)
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    m = mode.lower()
    mode_id = 0 if m == "max" else 1
    sptr = int(get_stream_ptr(stream))

    if _has("backward_ex"):
        _g.backward_ex(tX, tY, tgY, t_dX, kH, kW, sH, sW, pH, pW, mode_id, sptr)
        return dx
    if _has("backward"):
        _g.backward(tX, tY, tgY, t_dX, kH, kW, sH, sW, pH, pW, mode_id, sptr)
        return dx
    raise NotImplementedError("[ops.pool2d] 바인딩에 backward_ex/backward 함수가 없습니다.")
