# python/graph_executor_v2/ops/conv2d.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # 바인딩(새 시그니처):
    # forward(x_ptr,x_shape,w_ptr,w_shape,y_ptr,y_shape,bias_ptr|None,z_ptr|None, attrs, stream,
    #         dCol_ptr, W_KC_ptr, Y_tmp_ptr, Z_rows_ptr)
    # backward(x_ptr,x_shape,w_ptr,w_shape,dy_ptr,dy_shape,z_ptr,z_shape, dw_ptr|None, db_ptr|None, dx_ptr|None, attrs, stream,
    #          dCol_ptr, dTmp_ptr, W_CK_ptr, dWpack_ptr, dY_HT_ptr, gy_rows_ptr, Z_rows_ptr)
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
            Z_saved = out  # alias
        else:
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

    # ====== Workspaces (캡처-세이프) ======
    # K, HWo, 내부 행렬 크기 계산
    K   = (CinW * KH * KW)
    HWo = (H_out * W_out)

    # 필수
    dCol   = cp.empty((HWo, K),     dtype=cp.float32)  # [HWo,K]
    W_KC   = cp.empty((K,   Cout),  dtype=cp.float32)  # [K,Cout]
    Y_tmp  = cp.empty((HWo, Cout),  dtype=cp.float32)  # [HWo,Cout]

    # save_z면 행단위 Z 버퍼도 필요 (런처가 전치해 채운 뒤 NCHW로 복원)
    Z_rows = cp.empty((HWo, Cout), dtype=cp.float32) if save_z else None

    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_ptr,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        # workspaces
        int(dCol.data.ptr),
        int(W_KC.data.ptr),
        int(Y_tmp.data.ptr),
        int(Z_rows.data.ptr) if Z_rows is not None else 0
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

    # ====== Workspaces (캡처-세이프) ======
    K   = (CinW * KH * KW)
    H_out = Hy
    W_out = Wy
    HWo = (H_out * W_out)

    dCol   = cp.empty((HWo, K), dtype=cp.float32)          # 필수
    dTmp   = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)  # 필수(1D 임시)
    W_CK   = cp.empty((Cout, K), dtype=cp.float32) if want_gX else None
    dY_HT  = cp.empty((HWo, Cout), dtype=cp.float32) if want_gX else None
    dWpack = cp.empty((Cout, K), dtype=cp.float32) if want_gW else None
    gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)  # 필수
    Z_rows  = cp.empty((Cout, HWo), dtype=cp.float32)  # 필수

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(Z.data.ptr),  [Nz, Coz, Hz, Wz],
        int(gW.data.ptr) if gW is not None else None,
        int(gB.data.ptr) if gB is not None else None,
        int(gX.data.ptr) if gX is not None else None,
        attrs, sptr,
        # workspaces
        int(dCol.data.ptr),
        int(dTmp.data.ptr),
        int(W_CK.data.ptr)   if W_CK   is not None else 0,
        int(dWpack.data.ptr) if dWpack is not None else 0,
        int(dY_HT.data.ptr)  if dY_HT  is not None else 0,
        int(gy_rows.data.ptr),
        int(Z_rows.data.ptr)
    )

    out: Dict[str, cp.ndarray] = {}
    if gX is not None: out["gX"] = gX
    if gW is not None: out["gW"] = gW
    if gB is not None: out["gB"] = gB
    return out

class Conv2DWorkspaces:
    """
    캡처-세이프 워크스페이스 묶음.
    forward에 필요한: dCol, W_KC, Y_tmp, (옵션) Z_rows
    backward에 필요한: dCol, dTmp, gy_rows, Z_rows, (옵션) W_CK, dWpack, dY_HT
    """
    def __init__(self):
        # fwd
        self.dCol   = None  # [HWo, K]
        self.W_KC   = None  # [K, Cout]
        self.Y_tmp  = None  # [HWo, Cout]
        self.Z_rows = None  # [HWo, Cout] if save_z

        # bwd (공통/필수)
        self.dCol_b = None  # [HWo, K]
        self.dTmp   = None  # [max(Cout*K, HWo*K)]
        self.gy_rows= None  # [Cout, HWo]
        self.Z_rows_b=None  # [Cout, HWo]

        # bwd (옵션)
        self.W_CK   = None  # [Cout, K]    if want_gX
        self.dWpack = None  # [Cout, K]    if want_gW
        self.dY_HT  = None  # [HWo, Cout]  if want_gX

    def ensure_forward(self, *, HWo: int, K: int, Cout: int, save_z: bool):
        # 아래는 shape만 체크 → 없거나 shape 다르면 ValueError로 막고,
        # "캡처 전"에 호출자가 직접 할당하도록 강제.
        def _chk(arr, shape, name):
            if arr is None or arr.shape != shape or arr.dtype != cp.float32:
                raise ValueError(f"[capture] workspace `{name}` must be preallocated float32{shape}, "
                                 f"got {None if arr is None else (arr.shape, arr.dtype)}")

        _chk(self.dCol,  (HWo, K),     "dCol")
        _chk(self.W_KC,  (K,   Cout),  "W_KC")
        _chk(self.Y_tmp, (HWo, Cout),  "Y_tmp")
        if save_z:
            _chk(self.Z_rows, (HWo, Cout), "Z_rows")
        else:
            if self.Z_rows is not None:
                raise ValueError("[capture] Z_rows must be None when save_z=False")

    def ensure_backward(self, *, HWo: int, K: int, Cout: int,
                        want_gX: bool, want_gW: bool):
        def _chk(arr, shape, name):
            if arr is None or arr.shape != shape or arr.dtype != cp.float32:
                raise ValueError(f"[capture] workspace `{name}` must be preallocated float32{shape}, "
                                 f"got {None if arr is None else (arr.shape, arr.dtype)}")

        _chk(self.dCol_b, (HWo, K), "dCol_b")
        _chk(self.dTmp,   (max(Cout*K, HWo*K),), "dTmp")
        _chk(self.gy_rows,(Cout, HWo), "gy_rows")
        _chk(self.Z_rows_b,(Cout, HWo), "Z_rows_b")

        if want_gX:
            _chk(self.W_CK,  (Cout, K),   "W_CK")
            _chk(self.dY_HT, (HWo,  Cout),"dY_HT")
        else:
            if self.W_CK is not None or self.dY_HT is not None:
                raise ValueError("[capture] W_CK/dY_HT must be None when want_gX=False")

        if want_gW:
            _chk(self.dWpack, (Cout, K), "dWpack")
        else:
            if self.dWpack is not None:
                raise ValueError("[capture] dWpack must be None when want_gW=False")


def _out_hw(H: int, W: int, KH: int, KW: int,
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            dilation: Tuple[int, int]) -> Tuple[int, int]:
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
    return H_out, W_out


def forward_into(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW) (groups>1 -> Cin/groups)
    *,
    out: cp.ndarray,        # (N,Cout,H_out,W_out) preallocated
    B: Optional[cp.ndarray] = None,  # (Cout,)
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,
    Z_saved: Optional[cp.ndarray] = None,  # (N,Cout,H_out,W_out) preallocated if save_z
    stream: Optional[int] = None,
    work: Optional[Conv2DWorkspaces] = None,
) -> None:
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")
    if out is None or out.dtype != cp.float32 or out.ndim != 4:
        raise ValueError("[capture] `out` must be preallocated float32 4D")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    H_out, W_out = _out_hw(H, W_in, KH, KW, stride, padding, dilation)

    if tuple(out.shape) != (N, Cout, H_out, W_out):
        raise ValueError(f"[capture] out shape must be (N,Cout,H_out,W_out)={(N,Cout,H_out,W_out)}, "
                         f"got {tuple(out.shape)}")

    act_kind = _parse_act_kind(act)
    if save_z:
        if Z_saved is None or Z_saved.dtype != cp.float32 or tuple(Z_saved.shape)!=(N,Cout,H_out,W_out):
            raise ValueError(f"[capture] Z_saved must be preallocated float32[{(N,Cout,H_out,W_out)}] when save_z=True")
    else:
        if Z_saved is not None:
            raise ValueError("[capture] Z_saved must be None when save_z=False")

    if with_bias:
        if B is None or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != Cout:
            raise ValueError(f"[capture] B must be float32 1D length {Cout}")
        bias_ptr = int(B.data.ptr)
    else:
        bias_ptr = None

    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=save_z
    )
    sptr = int(get_stream_ptr(stream))

    # 내부 행렬 크기
    K   = (CinW * KH * KW)
    HWo = (H_out * W_out)

    if work is None:
        raise ValueError("[capture] provide `work: Conv2DWorkspaces` with preallocated buffers")
    work.ensure_forward(HWo=HWo, K=K, Cout=Cout, save_z=save_z)

    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_ptr,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        int(work.dCol.data.ptr),
        int(work.W_KC.data.ptr),
        int(work.Y_tmp.data.ptr),
        int(work.Z_rows.data.ptr) if save_z else 0,
    )
    # 반환 없음 (out/Z_saved에 in-place)


def backward_into(
    X: cp.ndarray,          # (N,Cin,H,W)
    W: cp.ndarray,          # (Cout,Cin,KH,KW)
    gY: cp.ndarray,         # (N,Cout,H_out,W_out)
    Z:  cp.ndarray,         # (N,Cout,H_out,W_out)
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    # 결과 버퍼들 (필요한 것만 제공; None이면 해당 그래디언트 스킵)
    gX_out: Optional[cp.ndarray] = None,     # (N,Cin,H,W)
    gW_out: Optional[cp.ndarray] = None,     # (Cout,Cin,KH,KW)
    gB_out: Optional[cp.ndarray] = None,     # (Cout,)
    stream: Optional[int] = None,
    work: Optional[Conv2DWorkspaces] = None,
) -> None:
    _assert_f32_4d(X, "X")
    _assert_f32_4d(W, "W")
    _assert_f32_4d(gY, "gY")
    _assert_f32_4d(Z,  "Z")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    Ny, Coy, Hy, Wy = map(int, gY.shape)
    Nz, Coz, Hz, Wz = map(int, Z.shape)

    if (Ny, Coy, Hy, Wy) != (N, Cout, Hz, Wz) or Nz != N or Coz != Cout:
        raise ValueError("[capture] gY/Z shapes must match (N,Cout,H_out,W_out) derived from X/W")

    if gX_out is not None and (gX_out.dtype != cp.float32 or tuple(gX_out.shape) != (N, Cin, H, W_in)):
        raise ValueError(f"[capture] gX_out must be float32[{(N,Cin,H,W_in)}]")
    if gW_out is not None and (gW_out.dtype != cp.float32 or tuple(gW_out.shape) != (Cout, CinW, KH, KW)):
        raise ValueError(f"[capture] gW_out must be float32[{(Cout,CinW,KH,KW)}]")
    if with_bias:
        if gB_out is None or gB_out.dtype != cp.float32 or gB_out.ndim != 1 or int(gB_out.size) != Cout:
            raise ValueError(f"[capture] gB_out must be float32 1D length {Cout} when with_bias=True")
    else:
        if gB_out is not None:
            raise ValueError("[capture] gB_out must be None when with_bias=False")

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=False
    )
    sptr = int(get_stream_ptr(stream))

    H_out, W_out = Hz, Wz
    HWo = (H_out * W_out)
    K   = (CinW * KH * KW)

    if work is None:
        raise ValueError("[capture] provide `work: Conv2DWorkspaces` with preallocated buffers")
    want_gX = gX_out is not None
    want_gW = gW_out is not None
    work.ensure_backward(HWo=HWo, K=K, Cout=Cout, want_gX=want_gX, want_gW=want_gW)

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(Z.data.ptr),  [Nz, Coz, Hz, Wz],
        int(gW_out.data.ptr) if gW_out is not None else None,
        int(gB_out.data.ptr) if gB_out is not None else None,
        int(gX_out.data.ptr) if gX_out is not None else None,
        attrs, sptr,
        int(work.dCol_b.data.ptr),
        int(work.dTmp.data.ptr),
        int(work.W_CK.data.ptr)   if want_gX else 0,
        int(work.dWpack.data.ptr) if want_gW else 0,
        int(work.dY_HT.data.ptr)  if want_gX else 0,
        int(work.gy_rows.data.ptr),
        int(work.Z_rows_b.data.ptr),
    )
