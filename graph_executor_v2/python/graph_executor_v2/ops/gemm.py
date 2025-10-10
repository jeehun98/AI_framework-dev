# python/graph_executor_v2/ops/gemm.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, Union
import cupy as cp

from .common import (
    assert_f32_2d,
    as_tensor_2d,
    empty_like_2d,
    empty_2d,
    get_stream_ptr,
    ensure_cuda_dlls,
    to_voidp_capsule,
)

# 바인딩(공용 타입 re-export 포함). 필요시 _ops_common 자동 로드.
from graph_executor_v2.ops import _ops_gemm as g

# (Windows) CUDA DLL 경로 가드
ensure_cuda_dlls()

# ============================================================
# Workspace helpers
# ============================================================
class GemmWorkspaces:
    """
    Capture-safe workspaces for GEMM backward.
      - dZ    : (M, N) float32, dAct(Z) ⊙ gY 임시 버퍼 (필수)
      - lt_ws : uint8 1D, cublasLt workspace (옵션)
    """
    def __init__(self):
        self.dZ: Optional[cp.ndarray] = None    # cp.ndarray(float32)[M,N]
        self.lt_ws: Optional[cp.ndarray] = None # cp.ndarray(uint8)[bytes] or None

    def ensure_backward(self, M: int, N: int):
        if self.dZ is None or self.dZ.dtype != cp.float32 or tuple(self.dZ.shape) != (M, N):
            raise ValueError(f"[capture] work.dZ must be preallocated float32[{M},{N}]")
        if self.lt_ws is not None and self.lt_ws.dtype != cp.uint8:
            raise ValueError("[capture] work.lt_ws must be uint8 or None")


_WS_CACHE: Dict[Tuple[int, int, int], GemmWorkspaces] = {}  # key: (M, N, dev_id)

def ensure_workspaces(M: int, N: int, *, lt_bytes: int = 0) -> GemmWorkspaces:
    """
    (M,N) + current device 기준으로 GemmWorkspaces를 캐시/재사용.
    캡처 전에 한 번 호출해두고 backward_into에 공급할 것.
    """
    dev = int(cp.cuda.runtime.getDevice())
    key = (M, N, dev)
    ws = _WS_CACHE.get(key)
    if ws is None:
        ws = GemmWorkspaces()
        ws.dZ = cp.empty((M, N), dtype=cp.float32)
        ws.lt_ws = cp.empty(lt_bytes, dtype=cp.uint8) if lt_bytes > 0 else None
        _WS_CACHE[key] = ws
    else:
        if ws.dZ is None or ws.dZ.shape != (M, N) or ws.dZ.dtype != cp.float32:
            ws.dZ = cp.empty((M, N), dtype=cp.float32)
        if lt_bytes > 0 and (ws.lt_ws is None or ws.lt_ws.nbytes < lt_bytes or ws.lt_ws.dtype != cp.uint8):
            ws.lt_ws = cp.empty(lt_bytes, dtype=cp.uint8)
    return ws

def clear_ws_cache() -> None:
    """워크스페이스 캐시 전체 제거(메모리 반환 유도)."""
    _WS_CACHE.clear()


# ============================================================
# Act / Attrs helpers
# ============================================================
def _parse_act_to_kind(act: str) -> "g.ActKind":
    # 대소/공백/하이픈/언더스코어 제거 후 비교
    s = (act or "none").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if s == "none":      return getattr(g.ActKind, "None")
    if s == "relu":      return g.ActKind.ReLU
    if s in ("leakyrelu", "lrelu"): return g.ActKind.LeakyReLU
    if s == "gelu":      return g.ActKind.GELU
    if s == "sigmoid":   return g.ActKind.Sigmoid
    if s == "tanh":      return g.ActKind.Tanh
    raise ValueError(f"unknown act: {act}")

def _mk_attrs(act: str, with_bias: bool, leaky_slope: float) -> "g.GemmAttrs":
    a = g.GemmAttrs()
    a.trans_a = False
    a.trans_b = False
    a.act = _parse_act_to_kind(act)  # enum 고정
    a.with_bias = bool(with_bias)
    a.leaky_slope = float(leaky_slope)
    # save_z는 attrs에 두지 않음 (포함된 변형도 가능하지만 여기선 호출부 인자로 처리)
    return a

def _has(name: str) -> bool:
    return hasattr(g, name)


# ============================================================
# Forward (allocating) : fused GEMM(+bias+activation)
# ============================================================
def forward(
    A: cp.ndarray,
    B: cp.ndarray,
    bias: Optional[cp.ndarray] = None,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    stream: Optional[int] = None,
    out: Optional[cp.ndarray] = None,
    # pre-activation Z 저장 옵션
    save_z: bool = False,
    z_out: Optional[cp.ndarray] = None,
    return_z: bool = False,
) -> Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]:
    """
    Fused GEMM(+bias+activation):  Y = act( A @ B (+ bias) )
      - A: (M, K), B: (K, N)
      - bias: (N,) | (1, N) | (M, 1) | (M, N) | scalar | None   (※ capture-safe 경로 아님)
      - dtype=float32, row-major 2D
      - save_z=True면 pre-activation Z를 z_out에 저장(없으면 내부에서 할당)
      - return_z=True면 (Y, Z)를 반환
    """
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")

    # with_bias 추론/검증
    if bias is not None and with_bias is False:
        raise ValueError("bias is provided but with_bias=False; set with_bias=True or remove bias")
    if with_bias is False and bias is not None:
        with_bias = True

    # bias 처리 (+ scalar 허용)  — allocating 경로에서는 친절히 보정 허용
    tBias = None
    if with_bias:
        if bias is None:
            raise ValueError("with_bias=True but bias is None")
        if cp.isscalar(bias):
            bias = cp.asarray(bias, dtype=cp.float32).reshape(1, 1)
        elif isinstance(bias, cp.ndarray):
            if bias.dtype != cp.float32:
                bias = bias.astype(cp.float32, copy=False)
            if bias.ndim == 1:
                if bias.size != N:
                    raise ValueError(f"bias 1D length must be N={N}, got {bias.size}")
                bias = bias.reshape(1, N)
            elif bias.ndim == 2:
                bm, bn = bias.shape
                # (1,N)|(M,1)|(M,N) 허용
                if not ((bm == 1 and bn == N) or (bm == M and bn == 1) or (bm == M and bn == N)):
                    raise ValueError(f"unsupported bias shape {bias.shape}; expected (1,N)|(M,1)|(M,N)")
            else:
                raise ValueError(f"unsupported bias ndim={bias.ndim}")
        else:
            raise TypeError("bias must be scalar or CuPy ndarray")
        tBias = as_tensor_2d(bias)

    # out 준비
    if out is None:
        out = cp.empty((M, N), dtype=cp.float32)
    elif out.shape != (M, N) or out.dtype != cp.float32:
        raise ValueError(f"out must be float32[{M},{N}]")

    # act='none'이면 Z==Y alias 가능
    act_is_none = (act or "none").strip().lower() in ("none",)

    # Z 저장 버퍼
    if save_z or return_z:
        if z_out is None:
            z_out = out if act_is_none else cp.empty((M, N), dtype=cp.float32)
        else:
            if z_out.shape != (M, N) or z_out.dtype != cp.float32:
                raise ValueError(f"z_out must be float32[{M},{N}]")
    else:
        z_out = None

    # 래핑
    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(out)
    tZ = as_tensor_2d(z_out) if z_out is not None else None

    stream_ptr = get_stream_ptr(stream)

    # ---- 호출: attrs 경로가 있으면 우선 사용, 없으면 구버전 시그니처로 폴백 ----
    if _has("forward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        # 바인딩은 Z_saved가 주어지면 내부에서 save_z를 암시적으로 활성화
        # (가능하면 attrs.save_z도 맞춰 주자 — 없는 구현일 수도 있으니 try/except)
        try:
            attrs.save_z = bool(save_z or return_z)
        except Exception:
            pass
        g.forward_ex_attrs(
            tA, tB, tBias, tY,
            attrs,
            tZ,
            to_voidp_capsule(stream_ptr),
        )
    else:
        # 구버전: 문자열 act/스칼라 인자 기반
        g.forward_ex(
            tA, tB, tBias, tY,
            False, False,                         # trans_a, trans_b
            act, bool(with_bias), float(leaky_slope),
            bool(save_z or return_z),             # save_z
            tZ,                                   # Z_saved
            to_voidp_capsule(stream_ptr),
        )

    if return_z or save_z:
        return out, z_out  # (Y, Z)
    return out


# ============================================================
# Backward (allocating)
# ============================================================
def backward(
    A: cp.ndarray,
    B: cp.ndarray,
    gY: cp.ndarray,
    Z: cp.ndarray,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    C: Optional[cp.ndarray] = None,   # if epilogue used C in forward (e.g., residual), else None
    want_gA: bool = True,
    want_gB: bool = True,
    want_gBias: bool = False,
    stream: Optional[int] = None,
    warn_mismatch: bool = False,
) -> Dict[str, cp.ndarray]:
    """
    Backward for fused GEMM(+bias+activation).
      Inputs : A(M,K), B(K,N), gY(M,N), Z(M,N)=pre-activation linear
               (optional) C(M,N) if used in forward epilogue
      Outputs: dict of { "gA", "gB", "gC", "gBias" } (요청된 것만 반환)

    Notes:
      - gBias는 PerN(=units) 축으로 반환되며 shape=(1, N)로 고정 생성합니다.
      - 평균(1/M)은 Loss가 책임. 레이어/커널은 합(sum)만 계산.
    """
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")
    Mg, Ng = assert_f32_2d(gY, "gY")
    Mz, Nz = assert_f32_2d(Z,  "Z")
    if Mg != M or Ng != N or Mz != M or Nz != N:
        raise ValueError(f"Shape mismatch: gY(Z) must be (M={M}, N={N})")

    if not isinstance(with_bias, bool):
        with_bias = bool(with_bias)
    if with_bias is False and want_gBias:
        raise ValueError("want_gBias=True requires with_bias=True")

    # 래핑
    tA  = as_tensor_2d(A)
    tB  = as_tensor_2d(B)
    tgY = as_tensor_2d(gY)
    tZ  = as_tensor_2d(Z)
    tC  = as_tensor_2d(C) if C is not None else None

    # 출력 버퍼 준비
    gA_arr = empty_like_2d(A) if want_gA else None
    gB_arr = empty_like_2d(B) if want_gB else None
    gC_arr = empty_like_2d(Z) if (C is not None) else None
    # ✅ PerN 보장: (1, N)
    gBias_arr = empty_2d(1, N) if (want_gBias and with_bias) else None

    t_gA    = as_tensor_2d(gA_arr)   if gA_arr  is not None else None
    t_gB    = as_tensor_2d(gB_arr)   if gB_arr  is not None else None
    t_gC    = as_tensor_2d(gC_arr)   if gC_arr  is not None else None
    t_gBias = as_tensor_2d(gBias_arr) if gBias_arr is not None else None

    stream_ptr = get_stream_ptr(stream)

    # ---- 호출: attrs 경로가 있으면 우선 사용, 없으면 구버전 시그니처로 폴백 ----
    if _has("backward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        g.backward_ex_attrs(
            tA, tB, tC, tgY, tZ,
            t_gA, t_gB, t_gC, t_gBias,
            attrs,
            to_voidp_capsule(stream_ptr),
        )
    else:
        g.backward_ex(
            tA, tB, tC, tgY, tZ,
            t_gA, t_gB, t_gC, t_gBias,
            False, False,
            act, bool(with_bias), float(leaky_slope),
            to_voidp_capsule(stream_ptr),
        )

    if warn_mismatch and gBias_arr is not None:
        if gBias_arr.shape != (1, N):
            print(f"[warn] gBias shape is {gBias_arr.shape}, expected (1,{N}). Ensure PerN bias grad (length N).")
        if not cp.isfinite(gBias_arr).all():
            print("[warn] gBias contains non-finite values.")

    out: Dict[str, cp.ndarray] = {}
    if gA_arr is not None:    out["gA"]    = gA_arr
    if gB_arr is not None:    out["gB"]    = gB_arr
    if gC_arr is not None:    out["gC"]    = gC_arr
    if gBias_arr is not None: out["gBias"] = gBias_arr
    return out


# ============================================================
# Forward (capture-safe, NO allocations)
# ============================================================
def forward_into(
    A: cp.ndarray,
    B: cp.ndarray,
    *,
    out: cp.ndarray,                   # (M,N) preallocated
    bias: Optional[cp.ndarray] = None, # ★ capture-safe: PerN (1,N)만 허용
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    save_z: bool = False,
    z_out: Optional[cp.ndarray] = None,   # (M,N) preallocated if save_z
    stream: Optional[int] = None,
) -> None:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")

    # --- STRICT no-allocation path ---
    if out is None or out.dtype != cp.float32 or out.shape != (M, N) or not out.flags.c_contiguous:
        raise ValueError(f"[capture] out must be C-contiguous float32[{M},{N}] (use cp.ascontiguousarray before capture)")
    if not (A.flags.c_contiguous and B.flags.c_contiguous):
        raise ValueError("[capture] inputs must be C-contiguous; use cp.ascontiguousarray before capture")

    if bias is not None and with_bias is False:
        raise ValueError("[capture] bias provided but with_bias=False; set with_bias=True or remove bias")

    tBias = None
    if with_bias:
        if bias is None:
            raise ValueError("[capture] with_bias=True but bias is None")
        # ★ capture-safe: PerN (1,N)만 허용 (브로드캐스트/astype/reshape 없음)
        if not (isinstance(bias, cp.ndarray) and bias.dtype == cp.float32 and bias.ndim == 2 and bias.shape == (1, N) and bias.flags.c_contiguous):
            raise ValueError("[capture] bias must be C-contiguous float32[1,N] (PerN) in capture-safe path")
        tBias = as_tensor_2d(bias)

    # act='none'이면 Z==Y alias 가능 (호출자가 동일 버퍼를 넘겨도 OK)
    act_is_none = (act or "none").strip().lower() in ("none",)

    if save_z:
        if z_out is None or z_out.dtype != cp.float32 or z_out.shape != (M, N) or not z_out.flags.c_contiguous:
            raise ValueError(f"[capture] z_out must be C-contiguous float32[{M},{N}] when save_z=True")
        # alias는 호출자가 동일 버퍼(out is z_out)로 넘기는 방식으로 사용 가능
    else:
        if z_out is not None:
            raise ValueError("[capture] z_out must be None when save_z=False")

    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(out)
    tZ = as_tensor_2d(z_out) if z_out is not None else (tY if (save_z and act_is_none) else None)
    stream_ptr = get_stream_ptr(stream)

    # attrs 우선, 폴백 허용
    if _has("forward_ex_attrs"):
        attrs = _mk_attrs(act, with_bias, leaky_slope)
        try:
            attrs.save_z = bool(save_z)
        except Exception:
            pass
        g.forward_ex_attrs(
            tA, tB, tBias, tY,
            attrs,
            tZ,
            to_voidp_capsule(stream_ptr),
        )
    else:
        g.forward_ex(
            tA, tB, tBias, tY,
            False, False,
            act, bool(with_bias), float(leaky_slope),
            bool(save_z),
            tZ,
            to_voidp_capsule(stream_ptr),
        )
    # return 없음 (in-place)


# ============================================================
# Backward (capture-safe, NO allocations)
# ============================================================
def backward_into(
    A: cp.ndarray,
    B: cp.ndarray,
    gY: cp.ndarray,
    Z: cp.ndarray,
    *,
    act: str = "none",
    with_bias: bool = False,
    leaky_slope: float = 0.01,
    C: Optional[cp.ndarray] = None,
    gA_out: Optional[cp.ndarray] = None,
    gB_out: Optional[cp.ndarray] = None,
    gC_out: Optional[cp.ndarray] = None,
    gBias_out: Optional[cp.ndarray] = None,
    stream: Optional[int] = None,
    # ★ 캡쳐-세이프용 워크스페이스 (필수: dZ, 옵션: lt_ws)
    work_dZ: Optional[cp.ndarray] = None,           # shape=(M,N) contiguous float32
    lt_workspace: Optional[cp.ndarray] = None,      # dtype=uint8 contiguous (e.g., 8MB)
) -> None:
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")
    Mg, Ng = assert_f32_2d(gY, "gY")
    Mz, Nz = assert_f32_2d(Z, "Z")
    if (Mg, Ng) != (M, N) or (Mz, Nz) != (M, N):
        raise ValueError(f"[capture] gY/Z must be (M={M}, N={N})")

    # 출력 버퍼 모양 검증(모두 contiguous)
    def _chk(arr, shape, name):
        if arr is None:
            return
        if arr.dtype != cp.float32 or tuple(arr.shape) != shape or not arr.flags.c_contiguous:
            raise ValueError(f"[capture] {name} must be C-contiguous float32{shape}")
    _chk(gA_out, (M, K), "gA_out")
    _chk(gB_out, (K, N), "gB_out")

    if C is None:
        if gC_out is not None:
            raise ValueError("[capture] gC_out must be None when C is None")
    else:
        Mc, Nc = assert_f32_2d(C, "C")
        if (Mc, Nc) != (M, N):
            raise ValueError(f"[capture] C must be (M={M}, N={N})")
        _chk(gC_out, (M, N), "gC_out")

    if with_bias:
        _chk(gBias_out, (1, N), "gBias_out")
    else:
        if gBias_out is not None:
            raise ValueError("[capture] gBias_out must be None when with_bias=False")

    # ★ 워크스페이스 필수: dZ
    if work_dZ is None or work_dZ.dtype != cp.float32 or work_dZ.size != M * N or not work_dZ.flags.c_contiguous:
        raise ValueError("[capture] work_dZ must be C-contiguous float32 with size M*N")
    dZ_ptr = int(work_dZ.data.ptr)

    # 옵션: Lt workspace
    if lt_workspace is not None:
        if lt_workspace.dtype != cp.uint8 or not lt_workspace.flags.c_contiguous:
            raise ValueError("[capture] lt_workspace must be C-contiguous uint8")
        lt_ptr   = int(lt_workspace.data.ptr)
        lt_bytes = int(lt_workspace.nbytes)
    else:
        lt_ptr, lt_bytes = 0, 0

    # 래핑
    tA  = as_tensor_2d(A)
    tB  = as_tensor_2d(B)
    tgY = as_tensor_2d(gY)
    tZ  = as_tensor_2d(Z)
    tC  = as_tensor_2d(C) if C is not None else None
    t_gA    = as_tensor_2d(gA_out)    if gA_out    is not None else None
    t_gB    = as_tensor_2d(gB_out)    if gB_out    is not None else None
    t_gC    = as_tensor_2d(gC_out)    if gC_out    is not None else None
    t_gBias = as_tensor_2d(gBias_out) if gBias_out is not None else None

    attrs = _mk_attrs(act, with_bias, leaky_slope)
    stream_ptr = get_stream_ptr(stream)

    if not _has("backward_into"):
        # 백엔드가 아직 capture-safe용 backward_into를 제공하지 않는다면,
        # 이 경로는 사용할 수 없음. (할당 없는 경로 요구사항 때문에)
        raise RuntimeError("g.backward_into entrypoint not found in bindings (capture-safe BWD unavailable)")

    # ★ 캡쳐-세이프 진입점 (C++ 바인딩 시그니처에 맞춤)
    g.backward_into(
        tA, tB, tC, tgY, tZ,
        t_gA, t_gB, t_gC, t_gBias,
        attrs,
        to_voidp_capsule(stream_ptr),
        int(dZ_ptr),
        int(lt_ptr),
        int(lt_bytes),
    )
