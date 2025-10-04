# python/graph_executor_v2/ops/gemm.py
from __future__ import annotations
from typing import Optional, Dict
import cupy as cp
from .common import (
    assert_f32_2d, as_tensor_2d, empty_like_2d, empty_2d,
    get_stream_ptr, ensure_cuda_dlls, to_voidp_capsule,   # ✅ 사용
)

# 바인딩(공용 타입 re-export 포함). 필요시 _ops_common 자동 로드.
from graph_executor_v2.ops import _ops_gemm as g

# (Windows) CUDA DLL 경로 가드
ensure_cuda_dlls()

# ------------------------------------------------------------
# Forward: fused GEMM(+bias+activation)
# ------------------------------------------------------------
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
) -> cp.ndarray:
    """
    Fused GEMM(+bias+activation):  Y = A @ B (+ bias) -> act
      - A: (M, K), B: (K, N)
      - bias: (1, N) | (M, 1) | (M, N) | None
      - dtype=float32, row-major 2D
    """
    M, K = assert_f32_2d(A, "A")
    K2, N = assert_f32_2d(B, "B")
    if K != K2:
        raise ValueError(f"K mismatch: A(K={K}) vs B(K={K2})")
    if with_bias and bias is None:
        raise ValueError("with_bias=True but bias is None")
    if bias is not None:
        assert_f32_2d(bias, "bias")

    if out is None:
        out = cp.empty((M, N), dtype=cp.float32)

    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(out)
    tBias = as_tensor_2d(bias) if bias is not None else None

    stream_ptr = get_stream_ptr(stream)

    g.forward_ex(
        tA, tB, tBias, tY,
        False, False,          # trans_a, trans_b
        act, with_bias, float(leaky_slope),
        to_voidp_capsule(stream_ptr)   # ✅ 캡슐 전달
    )
    return out


# ------------------------------------------------------------
# Backward: fused GEMM(+bias+activation)
#   - gBias는 반드시 "PerN" 축으로: shape=(1, N)
# ------------------------------------------------------------
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
    warn_mismatch: bool = False,      # ✅ 선택: 안전 경고 토글
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

    if with_bias is False and want_gBias:
        raise ValueError("want_gBias=True requires with_bias=True")

    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tgY = as_tensor_2d(gY)
    tZ = as_tensor_2d(Z)
    tC = as_tensor_2d(C) if C is not None else None

    # 출력 버퍼 준비
    gA_arr = empty_like_2d(A) if want_gA else None
    gB_arr = empty_like_2d(B) if want_gB else None
    gC_arr = empty_like_2d(Z) if (C is not None) else None

    # ✅ 핵심: gBias는 PerN 보장을 위해 (1, N)로 할당
    gBias_arr = empty_2d(1, N) if (want_gBias and with_bias) else None

    t_gA = as_tensor_2d(gA_arr) if gA_arr is not None else None
    t_gB = as_tensor_2d(gB_arr) if gB_arr is not None else None
    t_gC = as_tensor_2d(gC_arr) if gC_arr is not None else None
    t_gBias = as_tensor_2d(gBias_arr) if gBias_arr is not None else None

    stream_ptr = get_stream_ptr(stream)

    g.backward_ex(
        tA, tB, tC, tgY, tZ,
        t_gA, t_gB, t_gC, t_gBias,
        False, False,              # trans_a, trans_b
        act, with_bias, float(leaky_slope),
        to_voidp_capsule(stream_ptr)   # ✅ 캡슐 전달
    )

    # 선택: gBias 모양/NaN 방어 (디버그 시 유용)
    if warn_mismatch and gBias_arr is not None:
        if gBias_arr.shape != (1, N):
            # 이 경고가 뜨면 바인딩/런처에서 PerM으로 추론됐다는 뜻
            print(f"[warn] gBias shape is {gBias_arr.shape}, expected (1,{N}). "
                  f"Ensure PerN bias grad (length N).")

        # NaN/Inf 간단 체크
        if not cp.isfinite(gBias_arr).all():
            print("[warn] gBias contains non-finite values.")

    out: Dict[str, cp.ndarray] = {}
    if gA_arr is not None:    out["gA"]    = gA_arr
    if gB_arr is not None:    out["gB"]    = gB_arr
    if gC_arr is not None:    out["gC"]    = gC_arr
    if gBias_arr is not None: out["gBias"] = gBias_arr
    return out
