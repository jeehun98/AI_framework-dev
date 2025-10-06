from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # 바인딩(새 시그니처):
    # forward(x_ptr,x_shape,w_ptr,w_shape,y_ptr,y_shape,bias_ptr|None,z_ptr|None, attrs, stream)
    # backward(x_ptr,x_shape,w_ptr,w_shape,dy_ptr,dy_shape,z_ptr,z_shape, dw_ptr|None, db_ptr|None, dx_ptr|None, attrs, stream)
    from graph_executor_v2.ops import _ops_conv2d as _g
except Exception as e:
    raise ImportError(
        "[ops.conv2d] _ops_conv2d 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_conv2d)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# --------------------------- helpers ---------------------------
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
    *,
    with_bias: bool,
    act_kind: "_g.ActKind",
    leaky_slope: float,
    save_z: bool,
) -> _g.Conv2DAttrs:
    a = _g.Conv2DAttrs()
    a.stride_h, a.stride_w = int(stride[0]), int(stride[1])
    a.pad_h, a.pad_w       = int(padding[0]), int(padding[1])
    a.dil_h, a.dil_w       = int(dilation[0]), int(dilation[1])
    a.groups               = int(groups)
    a.with_bias            = bool(with_bias)
    a.act                  = act_kind
    a.leaky_slope          = float(leaky_slope)
    a.save_z               = bool(save_z)
    return a


def _parse_act_kind(act: str | "_g.ActKind") -> "_g.ActKind":
    if isinstance(act, _g.ActKind):
        return act
    s = (act or "none").lower()
    if s == "none":    return getattr(_g.ActKind, "None")
    if s == "relu":    return _g.ActKind.ReLU
    if s in ("leakyrelu", "leaky_relu", "lrelu"): return _g.ActKind.LeakyReLU
    if s == "gelu":    return _g.ActKind.GELU
    if s == "sigmoid": return _g.ActKind.Sigmoid
    if s == "tanh":    return _g.ActKind.Tanh
    raise ValueError(f"Unsupported act: {act}")


# --------------------------- forward ---------------------------
def forward(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW)  (groups>1이면 Cin/groups)
    B: Optional[cp.ndarray] = None,  # (Cout,)  (with_bias=True면 필수)
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,
    Z_saved: Optional[cp.ndarray] = None,  # (N,Cout,H_out,W_out). save_z=True면 필수(자동 생성 지원)
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Fused Conv2D(+bias+activation) forward.
      - X: (N,Cin,H,W), W: (Cout,Cin,KH,KW), B: (Cout,)
      - Y = act( conv(X,W) + b )
      - save_z=True면 pre-activation Z를 Z_saved에 저장 (Z_saved shape == Y)
        * act='none'이면 Z==Y라서 Z_saved를 out과 alias로 둬서 추가 메모리 없이 저장
    """
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)

    # 출력 크기
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W_in + 2*pW - dW*(KW-1) - 1)//sW + 1

    # out 버퍼
    if out is None:
        out = cp.empty((N, Cout, H_out, W_out), dtype=cp.float32)

    # act / attrs
    act_kind = _parse_act_kind(act)

    # --- save_z 자동 처리 ---
    # 사용자가 Z_saved를 주지 않았는데 save_z=True면 기본 버퍼를 마련
    if save_z and Z_saved is None:
        if act_kind == getattr(_g.ActKind, "None"):
            # 활성화 없음 → Z와 Y가 동일하므로 alias
            Z_saved = out
        else:
            # 활성화 있음 → Z가 Y로 덮어쓰이므로 별도 버퍼 필요
            Z_saved = cp.empty_like(out)

    if Z_saved is not None and Z_saved.shape != (N, Cout, H_out, W_out):
        raise ValueError(
            f"Z_saved must have shape (N,Cout,Hout,Wout)={(N, Cout, H_out, W_out)}, "
            f"got {tuple(Z_saved.shape)}"
        )

    # with_bias 체크 및 포인터 준비
    if with_bias:
        if B is None:
            raise ValueError("with_bias=True but B is None")
        if not (isinstance(B, cp.ndarray) and B.dtype == cp.float32 and B.ndim == 1 and int(B.size) == Cout):
            raise ValueError(f"B must be float32 1D of length Cout={Cout}")
        bias_ptr: Optional[int] = int(B.data.ptr)
    else:
        bias_ptr = None

    # attrs
    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=save_z
    )
    sptr = int(get_stream_ptr(stream))

    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_ptr,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr
    )
    return out


# --------------------------- backward ---------------------------
def backward(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW)
    gY: cp.ndarray,         # (N,Cout,H_out,W_out)
    Z:  cp.ndarray,         # (N,Cout,H_out,W_out)  pre-activation (필수)
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    want_gX: bool = True,
    want_gW: bool = True,
    want_gB: bool = False,
    stream: Optional[int] = None,
) -> Dict[str, cp.ndarray]:
    """
    Fused Conv2D(+bias+activation) backward.
      - 반드시 Z(pre-activation)가 필요 (dAct(Z) ⊙ gY 위해).
      - gB는 채널 기준 합(sum)로 반환: shape=(Cout,)
    """
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")
    _assert_f32_4d(gY, "gY")
    _assert_f32_4d(Z,  "Z")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    Ny, Coy, Hy, Wy = map(int, gY.shape)
    Nz, Coz, Hz, Wz = map(int, Z.shape)

    if Ny != N or Coy != Cout or Hy != Hz or Wy != Wz:
        raise ValueError("gY and Z must share (N,Cout,H_out,W_out) and match X/W derived output")
    if N != Nz or Cout != Coz:
        raise ValueError("Z must match N and Cout")

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=False  # bwd에서는 저장 불필요
    )
    sptr = int(get_stream_ptr(stream))

    # 선택적 출력 버퍼들
    gX = cp.empty_like(X)                if want_gX else None
    gW = cp.empty_like(W)                if want_gW else None
    gB = cp.empty((Cout,), cp.float32)   if (want_gB and with_bias) else None

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(Z.data.ptr),  [Nz, Coz, Hz, Wz],
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
