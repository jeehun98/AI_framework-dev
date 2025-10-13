# python/graph_executor_v2/ops/rnn.py
from __future__ import annotations
from typing import Optional, Dict, Tuple
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    # C++ 바인딩 (_ops_rnn): ActKind는 _ops_common 에서 재노출됨 (conv2d와 동일 패턴)
    from graph_executor_v2.ops import _ops_rnn as _g
except Exception as e:
    raise ImportError(
        "[ops.rnn] _ops_rnn 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_rnn)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# --------------------------- helpers ---------------------------
def _assert_f32_3d(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 3:
        raise ValueError(f"{name}: expected 3D, got shape={x.shape}")

def _assert_f32_2d(x: cp.ndarray, name: str):
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D, got shape={x.shape}")

def _assert_c_contig(*arrs: Tuple[str, cp.ndarray]):
    for name, a in arrs:
        if not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

def _act_none():
    for key in ("None", "NONE", "kNone"):
        if hasattr(_g.ActKind, key):
            return getattr(_g.ActKind, key)
    raise AttributeError("ActKind 'None' not found in binding")

def _parse_act_kind(act: str | "_g.ActKind") -> "_g.ActKind":
    if isinstance(act, _g.ActKind):
        return act
    s = (act or "none").lower().replace("_", "")
    if s == "none":    return _act_none()
    if s == "relu":    return _g.ActKind.ReLU
    if s in ("leakyrelu", "lrelu"): return _g.ActKind.LeakyReLU
    if s == "gelu":    return _g.ActKind.GELU
    if s == "sigmoid": return _g.ActKind.Sigmoid
    if s == "tanh":    return _g.ActKind.Tanh
    raise ValueError(f"Unsupported act: {act}")

def _attrs_from_args(
    *, with_bias: bool, act_kind: "_g.ActKind", leaky_slope: float, save_z: bool
) -> _g.RnnAttrs:
    a = _g.RnnAttrs()
    a.with_bias   = bool(with_bias)
    a.act         = act_kind
    a.leaky_slope = float(leaky_slope)
    a.save_z      = bool(save_z)
    return a


# --------------------------- public (non-capture) ---------------------------
def forward(
    X: cp.ndarray,          # (N,T,I)
    Wx: cp.ndarray,         # (I,H)
    Wh: cp.ndarray,         # (H,H)
    h0: cp.ndarray,         # (N,H)
    B: Optional[cp.ndarray] = None,  # (H,)
    *,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,
    Z_saved: Optional[cp.ndarray] = None,  # save_z=True면 (N,T,H); act=None이면 out과 alias 가능
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,      # (N,T,H)
) -> cp.ndarray:
    """
    Fused RNN(Elman) forward (간편/비캡처 경로).
    - 내부에서 WS 버퍼를 즉시 할당하므로 캡처에는 forward_into() 사용 권장.
    - [옵션] act=None이면 Z를 out과 alias 하여 저장(내부 save_z 강제 on).
    """
    # dtypes/shapes
    _assert_f32_3d(X, "X"); _assert_f32_2d(Wx, "Wx"); _assert_f32_2d(Wh, "Wh"); _assert_f32_2d(h0, "h0")
    if not X.flags.c_contiguous:  X = cp.ascontiguousarray(X)
    if not Wx.flags.c_contiguous: Wx = cp.ascontiguousarray(Wx)
    if not Wh.flags.c_contiguous: Wh = cp.ascontiguousarray(Wh)
    if not h0.flags.c_contiguous: h0 = cp.ascontiguousarray(h0)

    N, T, I = map(int, X.shape)
    Iw, H   = map(int, Wx.shape)
    Hh0     = int(h0.shape[1])
    if Iw != I:                                raise ValueError(f"Wx.shape[0]={Iw} must equal I={I}")
    if Wh.shape != (H, H):                     raise ValueError(f"Wh must be (H,H), got {Wh.shape}")
    if h0.shape != (N, H):                     raise ValueError(f"h0 must be (N,H), got {h0.shape}")

    if out is None:
        out = cp.empty((N, T, H), dtype=cp.float32)
    else:
        if out.dtype != cp.float32 or out.shape != (N, T, H):
            raise ValueError(f"out must be float32[{(N,T,H)}]")

    act_kind = _parse_act_kind(act)
    effective_save_z = bool(save_z or (act_kind == _act_none()))

    # Z_saved 처리
    if effective_save_z:
        if Z_saved is None:
            if act_kind == _act_none():
                Z_saved = out
            else:
                Z_saved = cp.empty_like(out)
        else:
            if Z_saved.dtype != cp.float32 or Z_saved.shape != (N, T, H):
                raise ValueError("Z_saved shape/dtype mismatch")
    else:
        if Z_saved is not None:
            raise ValueError("Z_saved must be None when save_z=False and act!=None")

    # bias
    if with_bias:
        if B is None or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != H:
            raise ValueError(f"B must be float32 1D len={H}")
        bias_obj = int(B.data.ptr)
    else:
        bias_obj = None

    attrs = _attrs_from_args(with_bias=with_bias, act_kind=act_kind,
                             leaky_slope=leaky_slope, save_z=effective_save_z)
    sptr = int(get_stream_ptr(stream))

    # ===== Workspaces =====
    XH_cat = cp.empty((N, I + H), dtype=cp.float32)
    Y_rows = cp.empty((N, H),     dtype=cp.float32)
    W_cat  = cp.empty((I + H, H), dtype=cp.float32)
    Z_rows = cp.empty((N, H),     dtype=cp.float32) if effective_save_z else None

    _g.forward(
        int(X.data.ptr),  [N, T, I],
        int(Wx.data.ptr), [I, H],
        int(Wh.data.ptr), [H, H],
        int(h0.data.ptr), [N, H],
        int(out.data.ptr), [N, T, H],
        bias_obj,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        int(XH_cat.data.ptr),
        int(Y_rows.data.ptr),
        int(W_cat.data.ptr),
        int(Z_rows.data.ptr) if Z_rows is not None else 0
    )
    return out


def backward(
    X: cp.ndarray, Wx: cp.ndarray, Wh: cp.ndarray, h0: cp.ndarray,
    dY_post: cp.ndarray, Z: cp.ndarray,
    *,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    want_dX: bool = True,
    want_dWx: bool = True,
    want_dWh: bool = True,
    want_dB: bool = False,      # with_bias=True일 때만 의미 있음
    want_dh0: bool = True,
    stream: Optional[int] = None,
) -> Dict[str, cp.ndarray]:
    """
    Fused RNN(Elman) backward (간편/비캡처 경로).
    - forward에서 save_z=True로 저장한 Z(pre-activation)가 반드시 필요합니다.
    """
    # contiguity & shapes
    for name, arr in (("X", X), ("dY_post", dY_post), ("Z", Z)):
        _assert_f32_3d(arr, name)
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    for name, arr in (("Wx", Wx), ("Wh", Wh), ("h0", h0)):
        _assert_f32_2d(arr, name)
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    N, T, I = map(int, X.shape)
    Ndy, Tdy, H = map(int, dY_post.shape)
    Nz, Tz, Hz = map(int, Z.shape)
    if (Ndy, Tdy, H) != (N, T, Hz) or Nz != N or Tz != T:
        raise ValueError("dY_post/Z shapes must be (N,T,H) and match X time/batch")

    if Wx.shape != (I, H):  raise ValueError(f"Wx must be (I,H), got {Wx.shape}")
    if Wh.shape != (H, H):  raise ValueError(f"Wh must be (H,H), got {Wh.shape}")
    if h0.shape != (N, H):  raise ValueError(f"h0 must be (N,H), got {h0.shape}")

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(with_bias=with_bias, act_kind=act_kind,
                             leaky_slope=leaky_slope, save_z=False)
    sptr = int(get_stream_ptr(stream))

    # 선택적 출력 준비
    dX  = cp.empty_like(X)          if want_dX  else None
    dWx = cp.empty_like(Wx)         if want_dWx else None
    dWh = cp.empty_like(Wh)         if want_dWh else None
    dB  = cp.empty((H,), cp.float32) if (want_dB and with_bias) else None
    dh0 = cp.empty_like(h0)         if want_dh0 else None

    # ===== Workspaces =====
    XH_cat  = cp.empty((N, I + H), dtype=cp.float32)
    G_rows  = cp.empty((N, H),     dtype=cp.float32)
    Z_rows  = cp.empty((N, H),     dtype=cp.float32)
    W_cat   = cp.empty((I + H, H), dtype=cp.float32)
    dXH_cat = cp.empty((N, I + H), dtype=cp.float32)
    dWcat   = cp.empty((I + H, H), dtype=cp.float32)
    TmpW    = cp.empty((I + H, H), dtype=cp.float32)

    _g.backward(
        int(X.data.ptr),   [N, T, I],
        int(Wx.data.ptr),  [I, H],
        int(Wh.data.ptr),  [H, H],
        int(h0.data.ptr),  [N, H],
        int(dY_post.data.ptr), [N, T, H],
        int(Z.data.ptr),   [N, T, H],
        int(dWx.data.ptr) if dWx is not None else None,
        int(dWh.data.ptr) if dWh is not None else None,
        int(dB.data.ptr)  if dB  is not None else None,
        int(dh0.data.ptr) if dh0 is not None else None,
        int(dX.data.ptr)  if dX  is not None else None,
        attrs, sptr,
        int(XH_cat.data.ptr),
        int(G_rows.data.ptr),
        int(Z_rows.data.ptr),
        int(W_cat.data.ptr),
        int(dXH_cat.data.ptr),
        int(dWcat.data.ptr),
        int(TmpW.data.ptr),
    )

    out: Dict[str, cp.ndarray] = {}
    if dX  is not None: out["dX"]  = dX
    if dWx is not None: out["dWx"] = dWx
    if dWh is not None: out["dWh"] = dWh
    if dB  is not None: out["dB"]  = dB
    if dh0 is not None: out["dh0"] = dh0
    return out


# --------------------------- capture-safe API ---------------------------
class RnnWorkspaces:
    """
    캡처-세이프 워크스페이스 묶음.
      forward:  XH_cat[N,I+H], Y_rows[N,H], W_cat[I+H,H], (opt) Z_rows[N,H]
      backward: XH_cat[N,I+H], G_rows[N,H], Z_rows[N,H], W_cat[I+H,H],
                dXH_cat[N,I+H], dWcat[I+H,H], TmpW[I+H,H]
    """
    def __init__(self):
        # forward
        self.XH_cat: Optional[cp.ndarray] = None
        self.Y_rows: Optional[cp.ndarray] = None
        self.W_cat:  Optional[cp.ndarray] = None
        self.Z_rows_f: Optional[cp.ndarray] = None  # forward Z_rows

        # backward
        self.XH_cat_b: Optional[cp.ndarray] = None
        self.G_rows:   Optional[cp.ndarray] = None
        self.Z_rows_b: Optional[cp.ndarray] = None
        self.W_cat_b:  Optional[cp.ndarray] = None
        self.dXH_cat:  Optional[cp.ndarray] = None
        self.dWcat:    Optional[cp.ndarray] = None
        self.TmpW:     Optional[cp.ndarray] = None

    def ensure_forward(self, *, N: int, I: int, H: int, save_z: bool):
        def _chk(arr, shape, name):
            if arr is None or arr.dtype != cp.float32 or tuple(arr.shape) != tuple(shape):
                raise ValueError(f"[capture] workspace `{name}` must be float32{shape}, "
                                 f"got {None if arr is None else (arr.shape, arr.dtype)}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"[capture] workspace `{name}` must be C-contiguous")
        _chk(self.XH_cat, (N, I+H), "XH_cat")
        _chk(self.Y_rows, (N, H),   "Y_rows")
        _chk(self.W_cat,  (I+H, H), "W_cat")
        if save_z:
            _chk(self.Z_rows_f, (N, H), "Z_rows_f")
        else:
            if self.Z_rows_f is not None:
                raise ValueError("[capture] Z_rows_f must be None when save_z=False")

    def ensure_backward(self, *, N: int, I: int, H: int):
        def _chk(arr, shape, name):
            if arr is None or arr.dtype != cp.float32 or tuple(arr.shape) != tuple(shape):
                raise ValueError(f"[capture] workspace `{name}` must be float32{shape}, "
                                 f"got {None if arr is None else (arr.shape, arr.dtype)}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"[capture] workspace `{name}` must be C-contiguous")
        _chk(self.XH_cat_b, (N, I+H), "XH_cat_b")
        _chk(self.G_rows,   (N, H),   "G_rows")
        _chk(self.Z_rows_b, (N, H),   "Z_rows_b")
        _chk(self.W_cat_b,  (I+H, H), "W_cat_b")
        _chk(self.dXH_cat,  (N, I+H), "dXH_cat")
        _chk(self.dWcat,    (I+H, H), "dWcat")
        _chk(self.TmpW,     (I+H, H), "TmpW")


def forward_into(
    X: cp.ndarray, Wx: cp.ndarray, Wh: cp.ndarray, h0: cp.ndarray, *,
    out: cp.ndarray,                      # (N,T,H) preallocated
    B: Optional[cp.ndarray] = None,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    save_z: bool = False,                # 사용자가 False여도 act=None이면 내부에서 True로 간주
    Z_saved: Optional[cp.ndarray] = None,  # preallocated if effective_save_z; act=None이면 out alias 허용
    stream: Optional[int] = None,
    work: Optional[RnnWorkspaces] = None,
) -> None:
    """
    [캡처 전용]
    - `effective_save_z = save_z or (act==none)`
    - act==none이면 Z_saved가 None이어도 out에 alias하여 포인터 고정 가능
    - 이 경우에도 work.Z_rows_f는 반드시 준비되어야 함 (save_z=True 경로와 동일)
    """
    for name, arr in (("X", X), ("Wx", Wx), ("Wh", Wh), ("h0", h0), ("out", out)):
        if arr.ndim == 3: _assert_f32_3d(arr, name)
        elif arr.ndim == 2: _assert_f32_2d(arr, name)
        else: raise ValueError(f"{name}: unexpected ndim={arr.ndim}")
    _assert_c_contig(("X", X), ("Wx", Wx), ("Wh", Wh), ("h0", h0), ("out", out))
    if B is not None and not B.flags.c_contiguous:
        raise ValueError("B must be C-contiguous")

    N, T, I = map(int, X.shape)
    Iw, H   = map(int, Wx.shape)
    if Iw != I:                                raise ValueError(f"Wx.shape[0]={Iw} must equal I={I}")
    if Wh.shape != (H, H):                     raise ValueError(f"Wh must be (H,H), got {Wh.shape}")
    if h0.shape != (N, H):                     raise ValueError(f"h0 must be (N,H), got {h0.shape}")
    if out.shape != (N, T, H):                 raise ValueError(f"out must be {(N,T,H)}")

    act_kind = _parse_act_kind(act)
    effective_save_z = bool(save_z or (act_kind == _act_none()))

    # Z_saved 준비
    if effective_save_z:
        if Z_saved is None:
            if act_kind == _act_none():
                Z_saved = out
            else:
                raise ValueError(f"[capture] Z_saved must be float32[{(N,T,H)}] when save_z=True")
        else:
            if Z_saved.dtype != cp.float32 or Z_saved.shape != (N, T, H):
                raise ValueError(f"[capture] Z_saved must be float32[{(N,T,H)}] when save_z=True")
    else:
        if Z_saved is not None:
            raise ValueError("[capture] Z_saved must be None when save_z=False and act!=None")

    # bias
    if with_bias:
        if B is None or B.dtype != cp.float32 or B.ndim != 1 or int(B.size) != H:
            raise ValueError(f"[capture] B must be float32 1D len={H}")
        bias_obj = int(B.data.ptr)
    else:
        bias_obj = None

    attrs = _attrs_from_args(with_bias=with_bias, act_kind=act_kind,
                             leaky_slope=leaky_slope, save_z=effective_save_z)
    sptr = int(get_stream_ptr(stream))

    if work is None:
        raise ValueError("[capture] provide `work: RnnWorkspaces` with preallocated buffers")
    work.ensure_forward(N=N, I=I, H=H, save_z=effective_save_z)

    _g.forward(
        int(X.data.ptr),  [N, T, I],
        int(Wx.data.ptr), [I, H],
        int(Wh.data.ptr), [H, H],
        int(h0.data.ptr), [N, H],
        int(out.data.ptr), [N, T, H],
        bias_obj,
        int(Z_saved.data.ptr) if Z_saved is not None else None,
        attrs, sptr,
        int(work.XH_cat.data.ptr),
        int(work.Y_rows.data.ptr),
        int(work.W_cat.data.ptr),
        int(work.Z_rows_f.data.ptr) if effective_save_z else 0
    )
    # return 없음 (out/Z_saved in-place)


def backward_into(
    X: cp.ndarray, Wx: cp.ndarray, Wh: cp.ndarray, h0: cp.ndarray,
    dY_post: cp.ndarray, Z: cp.ndarray, *,
    with_bias: bool = False,
    act: str | "_g.ActKind" = "none",
    leaky_slope: float = 0.01,
    dX_out: Optional[cp.ndarray] = None,   # None이면 스킵
    dWx_out: Optional[cp.ndarray] = None,  # None이면 스킵
    dWh_out: Optional[cp.ndarray] = None,  # None이면 스킵
    dB_out: Optional[cp.ndarray]  = None,  # with_bias=True일 때만 사용
    dh0_out: Optional[cp.ndarray] = None,  # None이면 스킵
    stream: Optional[int] = None,
    work: Optional[RnnWorkspaces] = None,
) -> None:
    """
    [캡처 전용] forward에서 저장한 Z(pre-activation)를 사용.
    요청된 출력만 계산(포인터 None이면 스킵).
    """
    for name, arr in (("X", X), ("dY_post", dY_post), ("Z", Z)):
        _assert_f32_3d(arr, name)
    for name, arr in (("Wx", Wx), ("Wh", Wh), ("h0", h0)):
        _assert_f32_2d(arr, name)
    _assert_c_contig(("X", X), ("Wx", Wx), ("Wh", Wh), ("h0", h0), ("dY_post", dY_post), ("Z", Z))

    N, T, I = map(int, X.shape)
    Ndy, Tdy, H = map(int, dY_post.shape)
    if (Ndy, Tdy) != (N, T): raise ValueError("dY_post must match (N,T,*) of X")
    if Z.shape != (N, T, H): raise ValueError(f"Z must be (N,T,H) with H={H}")
    if Wx.shape != (I, H):   raise ValueError(f"Wx must be (I,H)")
    if Wh.shape != (H, H):   raise ValueError(f"Wh must be (H,H)")
    if h0.shape != (N, H):   raise ValueError(f"h0 must be (N,H)")

    if dX_out is not None and (dX_out.dtype != cp.float32 or dX_out.shape != (N, T, I) or not dX_out.flags.c_contiguous):
        raise ValueError(f"[capture] dX_out must be float32[{(N,T,I)}] and C-contiguous")
    if dWx_out is not None and (dWx_out.dtype != cp.float32 or dWx_out.shape != (I, H) or not dWx_out.flags.c_contiguous):
        raise ValueError(f"[capture] dWx_out must be float32[{(I,H)}] and C-contiguous")
    if dWh_out is not None and (dWh_out.dtype != cp.float32 or dWh_out.shape != (H, H) or not dWh_out.flags.c_contiguous):
        raise ValueError(f"[capture] dWh_out must be float32[{(H,H)}] and C-contiguous")
    if dB_out is not None:
        if not with_bias:
            raise ValueError("[capture] dB_out must be None when with_bias=False")
        if dB_out.dtype != cp.float32 or dB_out.ndim != 1 or int(dB_out.size) != H or not dB_out.flags.c_contiguous:
            raise ValueError(f"[capture] dB_out must be float32 1D len={H} and C-contiguous")
    if dh0_out is not None and (dh0_out.dtype != cp.float32 or dh0_out.shape != (N, H) or not dh0_out.flags.c_contiguous):
        raise ValueError(f"[capture] dh0_out must be float32[{(N,H)}] and C-contiguous")

    if work is None:
        raise ValueError("[capture] provide `work: RnnWorkspaces` with preallocated buffers")
    work.ensure_backward(N=N, I=I, H=H)

    act_kind = _parse_act_kind(act)
    attrs = _attrs_from_args(with_bias=with_bias, act_kind=act_kind,
                             leaky_slope=leaky_slope, save_z=False)
    sptr = int(get_stream_ptr(stream))

    _g.backward(
        int(X.data.ptr),   [N, T, I],
        int(Wx.data.ptr),  [I, H],
        int(Wh.data.ptr),  [H, H],
        int(h0.data.ptr),  [N, H],
        int(dY_post.data.ptr), [N, T, H],
        int(Z.data.ptr),   [N, T, H],
        int(dWx_out.data.ptr) if dWx_out is not None else None,
        int(dWh_out.data.ptr) if dWh_out is not None else None,
        int(dB_out.data.ptr)  if (with_bias and dB_out is not None) else None,
        int(dh0_out.data.ptr) if dh0_out is not None else None,
        int(dX_out.data.ptr)  if dX_out is not None else None,
        attrs, sptr,
        int(work.XH_cat_b.data.ptr),
        int(work.G_rows.data.ptr),
        int(work.Z_rows_b.data.ptr),
        int(work.W_cat_b.data.ptr),
        int(work.dXH_cat.data.ptr),
        int(work.dWcat.data.ptr),
        int(work.TmpW.data.ptr),
    )
