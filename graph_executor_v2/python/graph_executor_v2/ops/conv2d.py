# python/graph_executor_v2/ops/conv2d.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr

ensure_cuda_dlls()

try:
    # C++ 바인딩 (_ops_conv2d): ActKind는 _ops_common 에서 재노출됨
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

def _act_none():
    for key in ("None", "NONE", "kNone"):
        if hasattr(_g.ActKind, key):
            return getattr(_g.ActKind, key)
    raise AttributeError("ActKind 'None' not found in binding")

def _parse_act_kind(act: str | "_g.ActKind") -> "_g.ActKind":
    if isinstance(act, _g.ActKind):
        return act
    s = (act or "none").lower()
    if s == "none":    return _act_none()
    if s == "relu":    return _g.ActKind.ReLU
    if s in ("leakyrelu", "leaky_relu", "lrelu"): return _g.ActKind.LeakyReLU
    if s == "gelu":    return _g.ActKind.GELU
    if s == "sigmoid": return _g.ActKind.Sigmoid
    if s == "tanh":    return _g.ActKind.Tanh
    raise ValueError(f"Unsupported act: {act}")

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
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    if sH <= 0 or sW <= 0: raise ValueError("stride must be > 0")
    if pH < 0 or pW < 0:   raise ValueError("padding must be >= 0")
    if dH <= 0 or dW <= 0: raise ValueError("dilation must be > 0")

    a = _g.Conv2DAttrs()
    a.stride_h, a.stride_w = sH, sW
    a.pad_h, a.pad_w       = pH, pW
    a.dil_h, a.dil_w       = dH, dW
    a.groups               = int(groups)
    a.with_bias            = bool(with_bias)
    a.act                  = act_kind
    a.leaky_slope          = float(leaky_slope)
    a.save_z               = bool(save_z)
    return a

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


# --------------------------- public (non-capture) ---------------------------
def forward(
    X: cp.ndarray,          # (N,Cin,H,W)   contiguous
    W: cp.ndarray,          # (Cout,CinW,KH,KW)  (groups>1이면 CinW=Cin/groups)
    B: Optional[cp.ndarray] = None,  # (Cout,)
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,
    Z_saved: Optional[cp.ndarray] = None,  # save_z=True면 (N,Cout,H_out,W_out); act=None이면 out과 alias 가능
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Fused Conv2D(+bias+activation) forward (간편/비캡처 경로).
    - 내부에서 WS를 즉시 할당하므로 캡처용으로는 forward_into() 사용 권장.
    """
    _assert_f32_4d(X, "X"); _assert_f32_4d(W, "W")
    if not X.flags.c_contiguous:  X = cp.ascontiguousarray(X)
    if not W.flags.c_contiguous:  W = cp.ascontiguousarray(W)

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)

    # groups 검증
    if groups < 1: raise ValueError("groups must be >= 1")
    if Cin % groups != 0:
        raise ValueError(f"Cin={Cin} must be divisible by groups={groups}")
    if CinW * groups != Cin:
        raise ValueError(f"W expects Cin/groups; got CinW={CinW}, groups={groups}, but Cin={Cin}")
    if Cout % groups != 0:
        raise ValueError(f"Cout={Cout} must be divisible by groups={groups}")

    H_out, W_out = _out_hw(H, W_in, KH, KW, stride, padding, dilation)
    if out is None:
        out = cp.empty((N, Cout, H_out, W_out), dtype=cp.float32)

    act_kind = _parse_act_kind(act)

    # save_z 처리 (act=None이면 Z==Y alias 가능)
    if save_z and Z_saved is None:
        if act_kind == _act_none():
            Z_saved = out
        else:
            Z_saved = cp.empty_like(out)
    if Z_saved is not None and tuple(Z_saved.shape) != (N, Cout, H_out, W_out):
        raise ValueError("Z_saved shape mismatch")

    # bias
    if with_bias:
        if B is None or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != Cout:
            raise ValueError(f"B must be float32 1D len={Cout}")
        bias_obj = int(B.data.ptr)
    else:
        bias_obj = None  # 바인딩이 None 허용

    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=save_z
    )
    sptr = int(get_stream_ptr(stream))

    # ====== Workspaces (바인딩 명세: HWo 기준) ======
    K   = CinW * KH * KW
    HWo = H_out * W_out
    dCol   = cp.empty((HWo, K),     dtype=cp.float32)
    W_KC   = cp.empty((K,   Cout),  dtype=cp.float32)
    Y_tmp  = cp.empty((HWo, Cout),  dtype=cp.float32)
    Z_rows = cp.empty((HWo, Cout),  dtype=cp.float32) if save_z else None

    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_obj,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        int(dCol.data.ptr),
        int(W_KC.data.ptr),
        int(Y_tmp.data.ptr),
        int(Z_rows.data.ptr) if Z_rows is not None else 0
    )
    return out


def backward(
    X: cp.ndarray, W: cp.ndarray, gY: cp.ndarray, Z: cp.ndarray,
    *, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1,
    with_bias=False, act: str | "_g.ActKind" = "none", leaky_slope=0.01,
    want_gX=True, want_gW=True, want_gB=False, stream: Optional[int] = None
) -> Dict[str, cp.ndarray]:
    """
    Fused Conv2D(+bias+activation) backward (간편/비캡처 경로).
    - 내부에서 WS를 즉시 할당하므로 캡처용으로는 backward_into() 사용 권장.
    """
    for name,t in (("X",X),("W",W),("gY",gY),("Z",Z)):
        _assert_f32_4d(t, name)
        if not t.flags.c_contiguous:
            locals()[name] = cp.ascontiguousarray(t)
    X,W,gY,Z = locals()["X"], locals()["W"], locals()["gY"], locals()["Z"]

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    Ny, Coy, Hy, Wy = map(int, gY.shape)
    Nz, Coz, Hz, Wz = map(int, Z.shape)

    if groups < 1: raise ValueError("groups must be >= 1")
    if Cin % groups != 0: raise ValueError("Cin % groups != 0")
    if CinW * groups != Cin: raise ValueError("W expects Cin/groups")
    if Cout % groups != 0: raise ValueError("Cout % groups != 0")

    if (Ny, Coy, Hy, Wy) != (N, Cout, Hz, Wz) or Nz != N or Coz != Cout:
        raise ValueError("gY/Z shapes must match (N,Cout,H_out,W_out)")

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=False
    )
    sptr = int(get_stream_ptr(stream))

    # 선택적 출력
    gX = cp.empty_like(X)              if want_gX else None
    gW = cp.empty_like(W)              if want_gW else None
    gB = cp.empty((Cout,), cp.float32) if (want_gB and with_bias) else None

    # ====== Workspaces (HWo 기준) ======
    K   = CinW * KH * KW
    HWo = Hy * Wy
    dCol   = cp.empty((HWo, K), dtype=cp.float32)
    dTmp   = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
    W_CK   = cp.empty((Cout, K), dtype=cp.float32) if want_gX else None
    dY_HT  = cp.empty((HWo,  Cout), dtype=cp.float32) if want_gX else None
    dWpack = cp.empty((Cout, K), dtype=cp.float32) if want_gW else None
    gy_rows= cp.empty((Cout, HWo), dtype=cp.float32)
    Z_rows = cp.empty((Cout, HWo), dtype=cp.float32)

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(Z.data.ptr),  [Nz, Coz, Hz, Wz],
        int(gW.data.ptr) if gW is not None else None,
        int(gB.data.ptr) if gB is not None else None,
        int(gX.data.ptr) if gX is not None else None,
        attrs, sptr,
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


# --------------------------- capture-safe API ---------------------------
class Conv2DWorkspaces:
    """
    캡처-세이프 워크스페이스 묶음 (바인딩 명세 준수: HWo 기준).
    forward:  dCol[HWo,K], W_KC[K,Cout], Y_tmp[HWo,Cout], (opt) Z_rows[HWo,Cout]
    backward: dCol[HWo,K], dTmp[max(Cout*K, HWo*K)], (opt) W_CK[Cout,K], dY_HT[HWo,Cout],
              (opt) dWpack[Cout,K], gy_rows[Cout,HWo], Z_rows[Cout,HWo]
    """
    def __init__(self):
        # forward
        self.dCol   = None
        self.W_KC   = None
        self.Y_tmp  = None
        self.Z_rows = None
        # backward (필수/공통)
        self.dCol_b  = None
        self.dTmp    = None
        self.gy_rows = None
        self.Z_rows_b= None
        # backward (옵션)
        self.W_CK   = None
        self.dWpack = None
        self.dY_HT  = None

    def ensure_forward(self, *, HWo: int, K: int, Cout: int, save_z: bool):
        def _chk(arr, shape, name):
            if arr is None or arr.dtype != cp.float32 or tuple(arr.shape) != tuple(shape):
                raise ValueError(f"[capture] workspace `{name}` must be float32{shape}, "
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
            if arr is None or arr.dtype != cp.float32 or tuple(arr.shape) != tuple(shape):
                raise ValueError(f"[capture] workspace `{name}` must be float32{shape}, "
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


def forward_into(
    X: cp.ndarray, W: cp.ndarray, *,
    out: cp.ndarray,                      # (N,Cout,H_out,W_out) preallocated
    B: Optional[cp.ndarray] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,
    Z_saved: Optional[cp.ndarray] = None,  # preallocated if save_z
    stream: Optional[int] = None,
    work: Optional[Conv2DWorkspaces] = None,
) -> None:
    _assert_f32_4d(X, "X"); _assert_f32_4d(W, "W")
    if out is None or out.dtype != cp.float32 or out.ndim != 4:
        raise ValueError("[capture] `out` must be preallocated float32 4D")
    if not (X.flags.c_contiguous and W.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError("[capture] inputs/outputs must be C-contiguous")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    if groups < 1: raise ValueError("groups must be >= 1")
    if Cin % groups != 0 or CinW * groups != Cin:
        raise ValueError("groups/Cin mismatch: W expects Cin/groups")
    if Cout % groups != 0:
        raise ValueError("Cout % groups != 0")

    H_out, W_out = _out_hw(H, W_in, KH, KW, stride, padding, dilation)
    if tuple(out.shape) != (N, Cout, H_out, W_out):
        raise ValueError(f"[capture] out must be {(N,Cout,H_out,W_out)}")

    act_kind = _parse_act_kind(act)
    if save_z:
        if Z_saved is None or Z_saved.dtype != cp.float32 or tuple(Z_saved.shape)!=(N,Cout,H_out,W_out):
            raise ValueError(f"[capture] Z_saved must be float32[{(N,Cout,H_out,W_out)}] when save_z=True")
    else:
        if Z_saved is not None:
            raise ValueError("[capture] Z_saved must be None when save_z=False")

    if with_bias:
        if B is None or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != Cout:
            raise ValueError(f"[capture] B must be float32 1D len={Cout}")
        bias_obj = int(B.data.ptr)
    else:
        bias_obj = None

    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=save_z
    )
    sptr = int(get_stream_ptr(stream))

    K   = CinW * KH * KW
    HWo = H_out * W_out
    if work is None:
        raise ValueError("[capture] provide `work: Conv2DWorkspaces` with preallocated buffers")
    work.ensure_forward(HWo=HWo, K=K, Cout=Cout, save_z=save_z)

    _g.forward(
        int(X.data.ptr), [N, Cin, H, W_in],
        int(W.data.ptr), [Cout, CinW, KH, KW],
        int(out.data.ptr), [N, Cout, H_out, W_out],
        bias_obj,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        int(work.dCol.data.ptr),
        int(work.W_KC.data.ptr),
        int(work.Y_tmp.data.ptr),
        int(work.Z_rows.data.ptr) if save_z else 0
    )
    # return 없음 (out/Z_saved in-place)


def backward_into(
    X: cp.ndarray, W: cp.ndarray, gY: cp.ndarray, Z: cp.ndarray, *,
    stride=(1,1), padding=(0,0), dilation=(1,1), groups=1,
    with_bias=False, act: str | "_g.ActKind" = "none", leaky_slope=0.01,
    gX_out: Optional[cp.ndarray] = None,
    gW_out: Optional[cp.ndarray] = None,
    gB_out: Optional[cp.ndarray] = None,   # with_bias=True여도 None이면 gB 스킵
    stream: Optional[int] = None,
    work: Optional[Conv2DWorkspaces] = None,
) -> None:
    for name,t in (("X",X),("W",W),("gY",gY),("Z",Z)):
        _assert_f32_4d(t, name)
        if not t.flags.c_contiguous:
            raise ValueError(f"[capture] {name} must be C-contiguous")
    if gX_out is not None and not gX_out.flags.c_contiguous:
        raise ValueError("[capture] gX_out must be C-contiguous")
    if gW_out is not None and not gW_out.flags.c_contiguous:
        raise ValueError("[capture] gW_out must be C-contiguous")

    N, Cin, H, W_in = map(int, X.shape)
    Cout, CinW, KH, KW = map(int, W.shape)
    Ny, Coy, Hy, Wy = map(int, gY.shape)
    Nz, Coz, Hz, Wz = map(int, Z.shape)

    if groups < 1: raise ValueError("groups must be >= 1")
    if Cin % groups != 0 or CinW * groups != Cin:
        raise ValueError("groups/Cin mismatch: W expects Cin/groups")
    if Cout % groups != 0:
        raise ValueError("Cout % groups != 0")

    if (Ny, Coy, Hy, Wy) != (N, Cout, Hz, Wz) or Nz != N or Coz != Cout:
        raise ValueError("[capture] gY/Z shapes must match (N,Cout,H_out,W_out)")

    # 요청된 그래디언트만 계산
    want_gX = gX_out is not None
    want_gW = gW_out is not None
    want_gB = (with_bias and gB_out is not None)

    if gX_out is not None and (gX_out.dtype != cp.float32 or tuple(gX_out.shape)!=(N,Cin,H,W_in)):
        raise ValueError(f"[capture] gX_out must be float32[{(N,Cin,H,W_in)}]")
    if gW_out is not None and (gW_out.dtype != cp.float32 or tuple(gW_out.shape)!=(Cout,CinW,KH,KW)):
        raise ValueError(f"[capture] gW_out must be float32[{(Cout,CinW,KH,KW)}]")
    if gB_out is not None:
        if not with_bias:
            raise ValueError("[capture] gB_out must be None when with_bias=False")
        if gB_out.dtype != cp.float32 or gB_out.ndim != 1 or int(gB_out.size)!=Cout:
            raise ValueError(f"[capture] gB_out must be float32 1D len={Cout}")

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(
        stride, padding, dilation, groups,
        with_bias=with_bias, act_kind=act_kind,
        leaky_slope=leaky_slope, save_z=False
    )
    sptr = int(get_stream_ptr(stream))

    K   = CinW * KH * KW
    HWo = Hy * Wy
    if work is None:
        raise ValueError("[capture] provide `work: Conv2DWorkspaces` with preallocated buffers")
    work.ensure_backward(HWo=HWo, K=K, Cout=Cout, want_gX=want_gX, want_gW=want_gW)

    _g.backward(
        int(X.data.ptr),  [N, Cin, H, W_in],
        int(W.data.ptr),  [Cout, CinW, KH, KW],
        int(gY.data.ptr), [Ny, Coy, Hy, Wy],
        int(Z.data.ptr),  [Nz, Coz, Hz, Wz],
        int(gW_out.data.ptr) if want_gW else None,
        int(gB_out.data.ptr) if want_gB else None,
        int(gX_out.data.ptr) if want_gX else None,
        attrs, sptr,
        int(work.dCol_b.data.ptr),
        int(work.dTmp.data.ptr),
        int(work.W_CK.data.ptr)   if want_gX else 0,
        int(work.dWpack.data.ptr) if want_gW else 0,
        int(work.dY_HT.data.ptr)  if want_gX else 0,
        int(work.gy_rows.data.ptr),
        int(work.Z_rows_b.data.ptr),
    )
