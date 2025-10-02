# python/graph_executor_v2/ops/conv2d.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # 바인딩: Conv2DAttrs, forward(x_ptr,x_shape,w_ptr,w_shape,y_ptr,y_shape,bias_ptr|None, attrs, stream)
    #         backward(x_ptr,x_shape,w_ptr,w_shape,dy_ptr,dy_shape, dw_ptr|None, db_ptr|None, dx_ptr|None, attrs, stream)
    from graph_executor_v2.ops import _ops_conv2d as _g
except Exception as e:
    raise ImportError(
        "[ops.conv2d] _ops_conv2d 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_conv2d)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


def _assert_f32_4d(x: cp.ndarray, name: str = "array"):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"{name}: expected 4D (N,C,H,W), got shape={x.shape}")


def _attrs_from_args(
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> _g.Conv2DAttrs:
    a = _g.Conv2DAttrs()
    a.stride_h, a.stride_w = int(stride[0]), int(stride[1])
    a.pad_h, a.pad_w       = int(padding[0]), int(padding[1])
    a.dil_h, a.dil_w       = int(dilation[0]), int(dilation[1])
    a.groups               = int(groups)
    return a


def forward(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW)  (groups>1이면 Cin/groups)
    B: Optional[cp.ndarray] = None,  # (Cout,) — 바인딩이 내부에서 길이만 사용
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    # act는 이 바인딩엔 없음. 필요하면 상위에서 별도의 활성화 레이어 사용.
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """포인터+shape를 직접 넘기는 conv2d forward 래퍼 (NCHW)."""
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)

    # groups 체크(선택)
    if CinW * groups != Cin:
        # 일반적으로 W.shape[1] == Cin//groups
        pass

    # 출력 크기
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W_in + 2*pW - dW*(KW-1) - 1)//sW + 1

    # out 버퍼 준비
    if out is None:
        out = cp.empty((N, Cout, H_out, W_out), dtype=cp.float32)

    # bias 포인터 (None 또는 uintptr_t)
    if B is not None:
        if not isinstance(B, cp.ndarray) or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != Cout:
            raise ValueError(f"B must be float32 1D of length Cout={Cout}")
        bias_ptr: Optional[int] = int(B.data.ptr)
    else:
        bias_ptr = None

    attrs = _attrs_from_args(stride, padding, dilation, groups)
    sptr  = int(get_stream_ptr(stream))

    # 포인터 + shape 벡터 전달
    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_ptr, attrs, sptr
    )
    return out


def backward(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW)
    gY: cp.ndarray,         # (N,Cout,H_out,W_out)
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = True,
    want_gX: bool = True,
    want_gW: bool = True,
    want_gB: bool = True,
    stream: Optional[int] = None,
) -> Dict[str, cp.ndarray]:
    """포인터+shape를 직접 넘기는 conv2d backward 래퍼. 반환: dict('gX','gW','gB')"""
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")
    _assert_f32_4d(gY, "gY")

    N, Cin, H, W_in      = map(int, X.shape)
    Cout, CinW, KH, KW   = map(int, W.shape)
    Ny, Coy, Hy, Wy      = map(int, gY.shape)
    if Ny != N or Coy != Cout:
        raise ValueError("gY's N/Cout must match X/W")

    attrs = _attrs_from_args(stride, padding, dilation, groups)
    sptr  = int(get_stream_ptr(stream))

    # 선택적 출력 버퍼들 준비
    gX = cp.empty_like(X)           if want_gX else None
    gW = cp.empty_like(W)           if want_gW else None
    gB = cp.empty((Cout,), cp.float32) if (want_gB and with_bias) else None

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(gW.data.ptr) if gW is not None else None,
        int(gB.data.ptr) if gB is not None else None,
        int(gX.data.ptr) if gX is not None else None,
        attrs, sptr
    )

    out: Dict[str, cp.ndarray] = {}
    if gX is not None: out["gX"] = gX
    if gW is not None: out["gW"] = gW
    if gB is not None: out["gB"] = gB
    return out
