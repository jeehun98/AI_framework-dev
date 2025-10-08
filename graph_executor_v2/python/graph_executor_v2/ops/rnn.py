from __future__ import annotations
from typing import Optional, Tuple, Union
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

# C++ 바인딩
try:
    from graph_executor_v2.ops import _ops_rnn as _g
except Exception as e:
    raise ImportError(
        "[ops.rnn] _ops_rnn 바인딩이 없습니다. CMake 타겟(_ops_rnn)을 빌드해 "
        "python/graph_executor_v2/ops 경로에 배치하세요."
    ) from e


# ---------- basic tensor helpers ----------
def _assert_f32_2d(a: cp.ndarray, name: str) -> Tuple[int, int]:
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(a)}")
    if a.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {a.dtype}")
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 2D, got shape={a.shape}")
    return int(a.shape[0]), int(a.shape[1])


def _as_tensor_2d(arr: cp.ndarray) -> "_g.Tensor":
    m, n = _assert_f32_2d(arr, "array")
    return _g.make_tensor_2d(int(arr.data.ptr), [m, n])


def _as_tensor_1d(arr: cp.ndarray) -> "_g.Tensor":
    if not isinstance(arr, cp.ndarray):
        raise TypeError(f"expected cupy.ndarray, got {type(arr)}")
    if arr.dtype != cp.float32:
        raise TypeError(f"expected float32, got {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"expected 1D, got shape={arr.shape}")
    return _g.make_tensor_1d(int(arr.data.ptr), [int(arr.shape[0])])


def _mk_attrs(T: int, B: int, I: int, H: int, save_z: bool) -> "_g.RNNAttrs":
    a = _g.RNNAttrs()
    a.T, a.B, a.I, a.H = int(T), int(B), int(I), int(H)
    a.save_z = bool(save_z)
    return a


# ---------- workspace helpers ----------
def make_ws_fwd_from_arrays(prez_all: cp.ndarray, tmp_h: cp.ndarray, tmp_z: cp.ndarray) -> _g.RNNWorkspaceFwd:
    _assert_f32_2d(prez_all, "prez_all")
    _assert_f32_2d(tmp_h, "tmp_h")
    _assert_f32_2d(tmp_z, "tmp_z")
    return _g.make_ws_fwd(int(prez_all.data.ptr), int(tmp_h.data.ptr), int(tmp_z.data.ptr))


def make_ws_bwd_from_arrays(dHsum: cp.ndarray, dh_next: cp.ndarray,
                            dZ_all: cp.ndarray, Hprev_all: cp.ndarray) -> _g.RNNWorkspaceBwd:
    _assert_f32_2d(dHsum, "dHsum")
    _assert_f32_2d(dh_next, "dh_next")
    _assert_f32_2d(dZ_all, "dZ_all")
    _assert_f32_2d(Hprev_all, "Hprev_all")
    return _g.make_ws_bwd(int(dHsum.data.ptr), int(dh_next.data.ptr),
                          int(dZ_all.data.ptr), int(Hprev_all.data.ptr))


def _coerce_ws_fwd(
    ws_fwd: Optional[Union[_g.RNNWorkspaceFwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]],
    T: int, B: int, H: int
) -> Tuple[_g.RNNWorkspaceFwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
    """
    ws_fwd 가 None이면 필요한 크기로 CuPy 배열을 생성해 WS를 만든다.
    반환: (ws_fwd_obj, (prez_all, tmp_h, tmp_z))  — 배열은 생명주기 유지를 위해 함께 반환.
    """
    TB = T * B
    if ws_fwd is None:
        prez_all = cp.empty((TB, H), dtype=cp.float32)  # [TB,H]
        tmp_h    = cp.empty((B,  H), dtype=cp.float32)  # [B,H]
        tmp_z    = cp.empty((B,  H), dtype=cp.float32)  # [B,H]
        return make_ws_fwd_from_arrays(prez_all, tmp_h, tmp_z), (prez_all, tmp_h, tmp_z)
    if isinstance(ws_fwd, tuple):
        if len(ws_fwd) != 3:
            raise ValueError("ws_fwd tuple must be (prez_all, tmp_h, tmp_z)")
        prez_all, tmp_h, tmp_z = ws_fwd
        return make_ws_fwd_from_arrays(prez_all, tmp_h, tmp_z), (prez_all, tmp_h, tmp_z)
    # 이미 pybind 객체
    return ws_fwd, (None, None, None)  # type: ignore


def _coerce_ws_bwd(
    ws_bwd: Optional[Union[_g.RNNWorkspaceBwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]]],
    T: int, B: int, H: int
) -> Tuple[_g.RNNWorkspaceBwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]]:
    """
    ws_bwd 가 None이면 필요한 크기로 CuPy 배열을 생성해 WS를 만든다.
    반환: (ws_bwd_obj, (dHsum, dh_next, dZ_all, Hprev_all))
    """
    TB = T * B
    if ws_bwd is None:
        dHsum     = cp.empty((B,  H), dtype=cp.float32)
        dh_next   = cp.empty((B,  H), dtype=cp.float32)
        dZ_all    = cp.empty((TB, H), dtype=cp.float32)
        Hprev_all = cp.empty((TB, H), dtype=cp.float32)
        return make_ws_bwd_from_arrays(dHsum, dh_next, dZ_all, Hprev_all), (dHsum, dh_next, dZ_all, Hprev_all)
    if isinstance(ws_bwd, tuple):
        if len(ws_bwd) != 4:
            raise ValueError("ws_bwd tuple must be (dHsum, dh_next, dZ_all, Hprev_all)")
        dHsum, dh_next, dZ_all, Hprev_all = ws_bwd
        return make_ws_bwd_from_arrays(dHsum, dh_next, dZ_all, Hprev_all), (dHsum, dh_next, dZ_all, Hprev_all)
    # 이미 pybind 객체
    return ws_bwd, (None, None, None, None)  # type: ignore


# ---------- public API ----------
def forward(
    X: cp.ndarray,          # [TB, I] (T*B, I)
    h0: cp.ndarray,         # [B, H]
    Wx: cp.ndarray,         # [I, H]
    Wh: cp.ndarray,         # [H, H]
    *,
    b: Optional[cp.ndarray] = None,    # [H]
    T: int,
    B: int,
    save_z: bool = True,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,              # Hout [TB, H]
    zbuf: Optional[cp.ndarray] = None,             # Z    [TB, H] (save_z=True에서 권장/필수)
    ws_fwd: Optional[Union[_g.RNNWorkspaceFwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]] = None,
) -> tuple[cp.ndarray, Optional[cp.ndarray]]:
    """Vanilla RNN(tanh) forward. 모든 텐서는 float32, row-major 연속."""
    TB, I = _assert_f32_2d(X, "X")
    if TB != T * B:
        raise ValueError(f"X rows(TB)={TB} must equal T*B={T*B}")
    B0, H = _assert_f32_2d(h0, "h0")
    if B0 != B:
        raise ValueError(f"h0 rows(B)={B0} must equal B={B}")

    I0, H0 = _assert_f32_2d(Wx, "Wx")
    if I0 != I:
        raise ValueError(f"Wx rows(I)={I0} must equal I={I}")
    H1, H2 = _assert_f32_2d(Wh, "Wh")
    if H1 != H or H2 != H:
        raise ValueError("Wh must be [H, H]")

    if b is not None:
        if not isinstance(b, cp.ndarray) or b.dtype != cp.float32 or b.ndim != 1 or b.shape[0] != H:
            raise ValueError("b must be float32 [H]")

    if out is None:
        out = cp.empty((TB, H), dtype=cp.float32)
    else:
        _assert_f32_2d(out, "out")
        if out.shape != (TB, H):
            raise ValueError(f"out shape must be {(TB, H)}")

    if save_z:
        if zbuf is None:
            zbuf = cp.empty((TB, H), dtype=cp.float32)
        else:
            _assert_f32_2d(zbuf, "zbuf")
            if zbuf.shape != (TB, H):
                raise ValueError(f"zbuf shape must be {(TB, H)}")
    else:
        # save_z=False면 zbuf는 사용하지 않음
        zbuf = None

    # workspaces
    ws_fwd_obj, _ws_keep = _coerce_ws_fwd(ws_fwd, T, B, H)

    attrs = _mk_attrs(T, B, I, H, save_z)
    sptr = int(get_stream_ptr(stream))

    tX = _as_tensor_2d(X)
    th0 = _as_tensor_2d(h0)
    tWx = _as_tensor_2d(Wx)
    tWh = _as_tensor_2d(Wh)
    tH  = _as_tensor_2d(out)

    tB  = _as_tensor_1d(b) if b is not None else None
    tZ  = _as_tensor_2d(zbuf) if (save_z and zbuf is not None) else None

    # 바인딩: ws_fwd 전달
    _g.rnn_forward(tX, th0, tWx, tWh, tB, tH, tZ, attrs, sptr, ws_fwd_obj)
    return out, (zbuf if save_z else None)


def backward(
    X: cp.ndarray,
    Hout: cp.ndarray,
    h0: cp.ndarray,
    Wx: cp.ndarray,
    Wh: cp.ndarray,
    dHout: cp.ndarray,
    *,
    Zbuf: Optional[cp.ndarray] = None,   # save_z=True로 fwd했다면 제공 권장(검사/일관성)
    T: int,
    B: int,
    stream: Optional[int] = None,
    dX_out: Optional[cp.ndarray] = None,
    dh0_out: Optional[cp.ndarray] = None,
    dWx_out: Optional[cp.ndarray] = None,
    dWh_out: Optional[cp.ndarray] = None,
    dB_out: Optional[cp.ndarray] = None,
    ws_bwd: Optional[Union[_g.RNNWorkspaceBwd, Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]]] = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    TB, I = _assert_f32_2d(X, "X")
    TBh, H = _assert_f32_2d(Hout, "Hout")
    if TB != T * B:
        raise ValueError(f"X rows(TB)={TB} must equal T*B={T*B}")
    if TBh != TB:
        raise ValueError(f"Hout first dim {TBh} must match X rows {TB}")

    B0, H0 = _assert_f32_2d(h0, "h0")
    if B0 != B:
        raise ValueError(f"h0 rows(B)={B0} must equal B={B}")

    WI, WH1 = _assert_f32_2d(Wx, "Wx")
    if WI != I or WH1 != H:
        raise ValueError(f"Wx must be shape {(I, H)}, got {Wx.shape}")

    WHr, WHc = _assert_f32_2d(Wh, "Wh")
    if WHr != H or WHc != H:
        raise ValueError(f"Wh must be shape {(H, H)}, got {Wh.shape}")

    dTB, dH = _assert_f32_2d(dHout, "dHout")
    if dTB != TB or dH != H:
        raise ValueError(f"dHout must be shape {(TB, H)}, got {dHout.shape}")

    if Zbuf is not None:
        zTB, zH = _assert_f32_2d(Zbuf, "Zbuf")
        if zTB != TB or zH != H:
            raise ValueError(f"Zbuf must be shape {(TB, H)}, got {Zbuf.shape}")

    # allocate / validate outs
    dX  = dX_out  if dX_out  is not None else cp.empty_like(X)
    _TB, _I = _assert_f32_2d(dX, "dX")
    if (_TB, _I) != (TB, I):
        raise ValueError(f"dX must be shape {(TB, I)}, got {dX.shape}")

    dh0 = dh0_out if dh0_out is not None else cp.empty_like(h0)
    _B1, _H1 = _assert_f32_2d(dh0, "dh0")
    if (_B1, _H1) != (B, H):
        raise ValueError(f"dh0 must be shape {(B, H)}, got {dh0.shape}")

    dWx = dWx_out if dWx_out is not None else cp.empty_like(Wx)
    _WI2, _WH2 = _assert_f32_2d(dWx, "dWx")
    if (_WI2, _WH2) != (I, H):
        raise ValueError(f"dWx must be shape {(I, H)}, got {dWx.shape}")

    dWh = dWh_out if dWh_out is not None else cp.empty_like(Wh)
    _WHr2, _WHc2 = _assert_f32_2d(dWh, "dWh")
    if (_WHr2, _WHc2) != (H, H):
        raise ValueError(f"dWh must be shape {(H, H)}, got {dWh.shape}")

    dB  = dB_out  if dB_out  is not None else cp.empty((H,), dtype=cp.float32)
    if not isinstance(dB, cp.ndarray) or dB.dtype != cp.float32 or dB.ndim != 1 or dB.size != H:
        raise ValueError(f"dB must be float32 [H]={H}, got {getattr(dB,'shape',None)} {getattr(dB,'dtype',None)}")

    # workspaces
    ws_bwd_obj, _wsb_keep = _coerce_ws_bwd(ws_bwd, T, B, H)

    # attrs: save_z는 Zbuf 제공 여부로 맞춰서 세팅(캡처 시 경로 고정)
    attrs = _mk_attrs(T, B, I, H, save_z=(Zbuf is not None))
    sptr = int(get_stream_ptr(stream))

    # wrap & call
    tX   = _as_tensor_2d(X)
    tH   = _as_tensor_2d(Hout)
    tZ   = _as_tensor_2d(Zbuf) if Zbuf is not None else None
    th0  = _as_tensor_2d(h0)
    tWx  = _as_tensor_2d(Wx)
    tWh  = _as_tensor_2d(Wh)
    tdH  = _as_tensor_2d(dHout)

    tdX  = _as_tensor_2d(dX)
    tdh0 = _as_tensor_2d(dh0)
    tdWx = _as_tensor_2d(dWx)
    tdWh = _as_tensor_2d(dWh)
    tdB  = _as_tensor_1d(dB)

    _g.rnn_backward(tX, tH, tZ, th0, tWx, tWh, tdH, tdX, tdh0, tdWx, tdWh, tdB, attrs, sptr, ws_bwd_obj)
    return dX, dh0, dWx, dWh, dB
