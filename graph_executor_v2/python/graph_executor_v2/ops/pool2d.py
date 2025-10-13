# python/graph_executor_v2/ops/pool2d.py
from __future__ import annotations
from typing import Optional, Tuple, Union
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

# --- 바인딩 모듈 로드 ---
try:
    from graph_executor_v2.ops import _ops_pool2d as _g
except Exception as e:
    raise ImportError(
        "[ops.pool2d] _ops_pool2d 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_pool2d)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# ---------- 로컬 유틸 ----------
def _ensure_f32_4d(a: cp.ndarray, name: str):
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요, got {type(a)}")
    if a.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요, got {a.dtype}")
    if a.ndim != 4:
        raise ValueError(f"{name}: 4D (N,C,H,W) 필요, got shape={a.shape}")

def _ensure_i32_4d(a: cp.ndarray, name: str):
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요, got {type(a)}")
    if a.dtype != cp.int32:
        raise TypeError(f"{name}: int32 필요, got {a.dtype}")
    if a.ndim != 4:
        raise ValueError(f"{name}: 4D (N,C,H,W) 필요, got shape={a.shape}")

def _c_contig(a: cp.ndarray) -> cp.ndarray:
    return a if a.flags.c_contiguous else cp.ascontiguousarray(a)

def _out_hw(H: int, W: int,
            kH: int, kW: int,
            sH: int, sW: int,
            pH: int, pW: int,
            dH: int, dW: int,
            ceil_mode: bool) -> Tuple[int, int]:
    effKH = (kH - 1) * dH + 1
    effKW = (kW - 1) * dW + 1
    aH = H + 2 * pH - effKH
    aW = W + 2 * pW - effKW
    if ceil_mode:
        # ceil((a + 1)/s) == (a + s) // s  이지만 C++ 구현과 일치시키기 위해 다음 형태 사용
        Ho = (aH >= 0) and ((aH + sH - 1) // sH + 1) or 0
        Wo = (aW >= 0) and ((aW + sW - 1) // sW + 1) or 0
    else:
        Ho = (aH >= 0) and (aH // sH + 1) or 0
        Wo = (aW >= 0) and (aW // sW + 1) or 0
    return max(0, int(Ho)), max(0, int(Wo))

def _mk_attrs(kernel, stride, padding, dilation, ceil_mode, count_include_pad) -> _g.Pool2DAttrs:
    kH, kW = map(int, kernel)
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    attrs = _g.Pool2DAttrs()
    attrs.kH, attrs.kW = kH, kW
    attrs.sH, attrs.sW = sH, sW
    attrs.pH, attrs.pW = pH, pW
    attrs.dH, attrs.dW = dH, dW
    attrs.ceil_mode = bool(ceil_mode)
    attrs.count_include_pad = bool(count_include_pad)
    return attrs


# ---------- Public API ----------
def forward(
    X: cp.ndarray,
    *,
    kernel: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (2, 2),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,   # avg 전용
    mode: str = "max",                 # "max" | "avg"
    return_indices: bool = False,      # max 전용
    ws_indices: Optional[cp.ndarray] = None,  # 외부 WS 버퍼(int32, NCHW, Y와 동일 shape)
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
    out: Optional[cp.ndarray] = None,
) -> Union[cp.ndarray, Tuple[cp.ndarray, Optional[cp.ndarray]]]:
    """
    Pool2D forward (CUDA, capture-safe).
      - X: (N,C,H,W) float32
      - mode: "max" | "avg"
      - return_indices: True면 MaxPool에서 Indices(N,C,Ho,Wo,int32)도 생성/반환
      - ws_indices: 캡처 시 내부 할당 없이 Indices 기록만 필요할 때 외부 버퍼 주입
    """
    _ensure_f32_4d(X, "X")
    X = _c_contig(X)
    N, C, H, W = map(int, X.shape)

    attrs = _mk_attrs(kernel, stride, padding, dilation, ceil_mode, count_include_pad)
    Ho, Wo = _out_hw(H, W, attrs.kH, attrs.kW, attrs.sH, attrs.sW, attrs.pH, attrs.pW, attrs.dH, attrs.dW, attrs.ceil_mode)

    if out is None:
        out = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
    else:
        _ensure_f32_4d(out, "out")
        if tuple(out.shape) != (N, C, Ho, Wo):
            raise ValueError(f"out.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(out.shape)}")
        out = _c_contig(out)

    sptr = int(get_stream_ptr(stream))
    x_ptr, y_ptr = int(X.data.ptr), int(out.data.ptr)
    x_shape, y_shape = [int(v) for v in X.shape], [int(v) for v in out.shape]

    m = mode.lower()
    if m == "max":
        # indices 텐서 생성/검증
        ind_tensor: Optional[cp.ndarray] = None
        ind_ptr = None
        if return_indices:
            ind_tensor = cp.empty((N, C, Ho, Wo), dtype=cp.int32)
            ind_ptr = int(ind_tensor.data.ptr)
        elif ws_indices is not None:
            _ensure_i32_4d(ws_indices, "ws_indices")
            if tuple(ws_indices.shape) != (N, C, Ho, Wo):
                raise ValueError(f"ws_indices.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(ws_indices.shape)}")
            ws_indices = _c_contig(ws_indices)

        ws_ptr = None if ws_indices is None else int(ws_indices.data.ptr)

        _g.maxpool2d_forward(
            x_ptr, x_shape,
            y_ptr, y_shape,
            ind_ptr if ind_ptr is not None else None,
            ws_ptr if ws_ptr is not None else None,
            attrs,
            sptr,
        )
        return (out, ind_tensor) if return_indices else out

    elif m == "avg":
        _g.avgpool2d_forward(
            x_ptr, x_shape,
            y_ptr, y_shape,
            None,   # ws_scratch_ptr (옵션, 현재 미사용)
            attrs,
            sptr,
        )
        return out
    else:
        raise ValueError("mode must be 'max' or 'avg'")


def backward(
    X: cp.ndarray,
    Y: cp.ndarray,
    gY: cp.ndarray,
    *,
    kernel: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (2, 2),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,   # avg 전용
    mode: str = "max",
    indices: Optional[cp.ndarray] = None,      # MaxPool BWD: 필수(혹은 ws_indices로 대체)
    ws_indices: Optional[cp.ndarray] = None,   # 외부 WS 버퍼 (int32, NCHW)
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
) -> cp.ndarray:
    """
    Pool2D backward → dX (CUDA, capture-safe).
      - MaxPool: indices 또는 ws_indices 중 하나 반드시 필요(현재 커널 재계산 미지원)
      - AvgPool: indices 미사용
      - 입력 X,Y는 형태 검증 및 호환성만 위해 받음(커널은 dY,dX만 사용)
    """
    _ensure_f32_4d(X, "X"); _ensure_f32_4d(Y, "Y"); _ensure_f32_4d(gY, "gY")
    X = _c_contig(X); Y = _c_contig(Y); gY = _c_contig(gY)

    N, C, H, W = map(int, X.shape)
    attrs = _mk_attrs(kernel, stride, padding, dilation, ceil_mode, count_include_pad)
    Ho, Wo = _out_hw(H, W, attrs.kH, attrs.kW, attrs.sH, attrs.sW, attrs.pH, attrs.pW, attrs.dH, attrs.dW, attrs.ceil_mode)

    # gY/Y 모양 검증
    if tuple(gY.shape) != (N, C, Ho, Wo):
        raise ValueError(f"gY.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(gY.shape)}")
    if tuple(Y.shape) != (N, C, Ho, Wo):
        raise ValueError(f"Y.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(Y.shape)}")

    dX = cp.empty_like(X)
    dX = _c_contig(dX)

    sptr = int(get_stream_ptr(stream))
    dy_ptr, dx_ptr = int(gY.data.ptr), int(dX.data.ptr)
    dy_shape, dx_shape = [int(v) for v in gY.shape], [int(v) for v in dX.shape]

    m = mode.lower()
    if m == "max":
        ind_ptr = None
        ws_ptr = None

        if indices is not None:
            _ensure_i32_4d(indices, "indices")
            if tuple(indices.shape) != (N, C, Ho, Wo):
                raise ValueError(f"indices.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(indices.shape)}")
            indices = _c_contig(indices)
            ind_ptr = int(indices.data.ptr)
        if ws_indices is not None:
            _ensure_i32_4d(ws_indices, "ws_indices")
            if tuple(ws_indices.shape) != (N, C, Ho, Wo):
                raise ValueError(f"ws_indices.shape mismatch: expected {(N,C,Ho,Wo)}, got {tuple(ws_indices.shape)}")
            ws_indices = _c_contig(ws_indices)
            ws_ptr = int(ws_indices.data.ptr)

        if ind_ptr is None and ws_ptr is None:
            raise ValueError("MaxPool backward requires either `indices` or `ws_indices`.")

        _g.maxpool2d_backward(
            dy_ptr, dy_shape,
            ind_ptr if ind_ptr is not None else None,
            dx_ptr, dx_shape,
            ws_ptr if ws_ptr is not None else None,
            None,   # ws_scratch_ptr (옵션)
            attrs,
            sptr,
        )
        return dX

    elif m == "avg":
        _g.avgpool2d_backward(
            dy_ptr, dy_shape,
            dx_ptr, dx_shape,
            None,   # ws_scratch_ptr (옵션)
            attrs,
            sptr,
        )
        return dX
    else:
        raise ValueError("mode must be 'max' or 'avg'")
