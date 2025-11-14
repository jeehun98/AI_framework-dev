import os
import sys

# repo 루트: .../graph_executor_v2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# python/ 디렉토리를 sys.path에 추가
PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import numpy as np
import cupy as cp

from graph_executor_v2.ops import _ops_gemm as gemm


# ============================================================
#  Activation (fwd/bwd) 레퍼런스 구현 (NumPy)
# ============================================================

def act_forward_np(z: np.ndarray, act: str, leaky_slope: float = 0.01) -> np.ndarray:
    act = act.lower()
    if act == "none":
        return z
    if act == "relu":
        return np.maximum(z, 0.0)
    if act in ("leakyrelu", "leaky_relu", "lrelu"):
        return np.where(z > 0.0, z, leaky_slope * z)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if act == "tanh":
        return np.tanh(z)
    if act == "gelu":
        # 표준 GELU (approx) 사용
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        from math import sqrt
        return 0.5 * z * (1.0 + np.erf(z / np.float32(sqrt(2.0))))
    raise ValueError(f"unsupported act: {act}")


def act_backward_np(z: np.ndarray,
                    gy: np.ndarray,
                    act: str,
                    leaky_slope: float = 0.01) -> np.ndarray:
    """gZ = dL/dZ = dL/dY * dY/dZ"""
    act = act.lower()
    if act == "none":
        return gy
    if act == "relu":
        mask = (z > 0.0).astype(z.dtype)
        return gy * mask
    if act in ("leakyrelu", "leaky_relu", "lrelu"):
        slope = np.ones_like(z, dtype=z.dtype)
        slope[z < 0.0] = leaky_slope
        return gy * slope
    if act == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-z))
        return gy * s * (1.0 - s)
    if act == "tanh":
        t = np.tanh(z)
        return gy * (1.0 - t * t)
    if act == "gelu":
        # GELU 도함수: 근사식 사용 (버전 따라 다를 수 있어 약간 오차 감안)
        from math import sqrt, pi
        x = z
        k = np.sqrt(2.0 / np.pi)
        c = 0.044715
        x3 = x * x * x
        tanh_arg = k * (x + c * x3)
        t = np.tanh(tanh_arg)
        dtanh = 1.0 - t * t
        term1 = 0.5 * (1.0 + t)
        term2 = 0.5 * x * dtanh * k * (1.0 + 3.0 * c * x * x)
        dgelu = term1 + term2
        return gy * dgelu
    raise ValueError(f"unsupported act: {act}")


# ============================================================
#  GEMM + bias + act 레퍼런스 (NumPy)
# ============================================================

def gemm_forward_ref(A_h: np.ndarray,
                     B_h: np.ndarray,
                     bias_h: np.ndarray | None,
                     act: str,
                     leaky_slope: float = 0.01):
    """
    Z = A @ B + bias
    Y = act(Z)
    bias: None or shape (1, N)
    """
    Z = A_h @ B_h  # (M,K) @ (K,N) -> (M,N)
    if bias_h is not None:
        # bias_h: (1,N) broadcast over rows
        Z = Z + bias_h.astype(A_h.dtype)
    Y = act_forward_np(Z, act, leaky_slope)
    return Y, Z


def gemm_backward_ref(A_h: np.ndarray,
                      B_h: np.ndarray,
                      bias_h: np.ndarray | None,
                      gY_h: np.ndarray,
                      Z_h: np.ndarray,
                      act: str,
                      leaky_slope: float = 0.01):
    """
    gZ = gY * act'(Z)
    gA = gZ @ B^T
    gB = A^T @ gZ
    gBias = sum over rows (PerN: (1,N))
    """
    gZ = act_backward_np(Z_h, gY_h, act, leaky_slope)  # (M,N)

    gA = gZ @ B_h.T           # (M,N) @ (N,K) -> (M,K)
    gB = A_h.T @ gZ           # (K,M) @ (M,N) -> (K,N)
    if bias_h is not None:
        gBias = gZ.sum(axis=0, keepdims=True)  # (1,N)
    else:
        gBias = None
    return gA, gB, gBias


# ============================================================
#  Raw 포인터 기반 GEMM 테스트
# ============================================================

def run_case_raw(m=32, k=64, n=16,
                 with_bias=True,
                 act="relu",
                 leaky_slope=0.01,
                 atol=1e-5, rtol=1e-4):
    print(f"[smoke-raw] m={m}, k={k}, n={n}, with_bias={with_bias}, act={act}")

    rng = np.random.default_rng(2025)

    # -------- host 데이터 (reference 계산용) --------
    A_h = rng.standard_normal((m, k), dtype=np.float32)
    B_h = rng.standard_normal((k, n), dtype=np.float32)
    gY_h = rng.standard_normal((m, n), dtype=np.float32)

    if with_bias:
        bias_h = rng.standard_normal((1, n), dtype=np.float32)
    else:
        bias_h = None

    # 레퍼런스 fwd/bwd
    Y_ref, Z_ref = gemm_forward_ref(A_h, B_h, bias_h, act, leaky_slope)
    gA_ref, gB_ref, gBias_ref = gemm_backward_ref(
        A_h, B_h, bias_h, gY_h, Z_ref, act, leaky_slope
    )

    # -------- GPU(CuPy) 버퍼 --------
    A_d = cp.asarray(A_h)
    B_d = cp.asarray(B_h)
    gY_d = cp.asarray(gY_h)

    Y_d = cp.empty((m, n), dtype=cp.float32)
    Z_d = cp.empty((m, n), dtype=cp.float32)   # forward_raw 에서 save_z=True 로 채우게 할 것

    gA_d = cp.empty((m, k), dtype=cp.float32)
    gB_d = cp.empty((k, n), dtype=cp.float32)

    if with_bias:
        Bias_d  = cp.asarray(bias_h)
        gBias_d = cp.empty((1, n), dtype=cp.float32)
        Bias_ptr  = int(Bias_d.data.ptr)
        gBias_ptr = int(gBias_d.data.ptr)
    else:
        Bias_d  = None
        gBias_d = None
        Bias_ptr  = 0
        gBias_ptr = 0

    # C / gC 는 사용하지 않으므로 0 포인터
    C_ptr  = 0
    gC_ptr = 0

    # -------- 포인터 추출 --------
    A_ptr  = int(A_d.data.ptr)
    B_ptr  = int(B_d.data.ptr)
    Y_ptr  = int(Y_d.data.ptr)
    Z_ptr  = int(Z_d.data.ptr)
    gY_ptr = int(gY_d.data.ptr)
    gA_ptr = int(gA_d.data.ptr)
    gB_ptr = int(gB_d.data.ptr)

    # -------------------------------------------------
    # Forward_raw: save_z=True → Z_d 에 pre-act(Z) 저장
    # -------------------------------------------------
    print("[smoke-raw] running gemm.forward_raw() ...")
    gemm.forward_raw(
        A_ptr,
        B_ptr,
        Bias_ptr,
        Y_ptr,
        m, k, n,
        False,  # trans_a
        False,  # trans_b
        act,
        with_bias,
        leaky_slope,
        True,   # save_z
        Z_ptr,
        None,   # stream
    )

    cp.cuda.runtime.deviceSynchronize()

    # -------------------------------------------------
    # Backward_raw
    # -------------------------------------------------
    print("[smoke-raw] running gemm.backward_raw() ...")
    gemm.backward_raw(
        A_ptr,
        B_ptr,
        C_ptr,
        gY_ptr,
        Z_ptr,
        gA_ptr,
        gB_ptr,
        gC_ptr,
        gBias_ptr,
        m, k, n,
        False,  # trans_a
        False,  # trans_b
        act,
        with_bias,
        leaky_slope,
        None,   # stream
    )

    cp.cuda.runtime.deviceSynchronize()

    # -------- 결과 가져오기 --------
    Y_out  = cp.asnumpy(Y_d)
    Z_out  = cp.asnumpy(Z_d)
    gA_out = cp.asnumpy(gA_d)
    gB_out = cp.asnumpy(gB_d)
    if with_bias:
        gBias_out = cp.asnumpy(gBias_d)
    else:
        gBias_out = None

    # -------- NaN 체크 --------
    print("[smoke-raw] Y_out shape:", Y_out.shape, "nan?", np.isnan(Y_out).any())
    print("[smoke-raw] gA_out shape:", gA_out.shape, "nan?", np.isnan(gA_out).any())
    print("[smoke-raw] gB_out shape:", gB_out.shape, "nan?", np.isnan(gB_out).any())
    if with_bias:
        print("[smoke-raw] gBias_out shape:", gBias_out.shape, "nan?", np.isnan(gBias_out).any())

    # -------- 레퍼런스와 diff --------
    max_err_Y   = float(np.max(np.abs(Y_out - Y_ref)))
    max_err_Z   = float(np.max(np.abs(Z_out - Z_ref)))
    max_err_gA  = float(np.max(np.abs(gA_out - gA_ref)))
    max_err_gB  = float(np.max(np.abs(gB_out - gB_ref)))
    if with_bias:
        max_err_gBias = float(np.max(np.abs(gBias_out - gBias_ref)))
    else:
        max_err_gBias = 0.0

    print(f"[check] max|Y_out - Y_ref|     = {max_err_Y:.3e}")
    print(f"[check] max|Z_out - Z_ref|     = {max_err_Z:.3e}")
    print(f"[check] max|gA_out - gA_ref|   = {max_err_gA:.3e}")
    print(f"[check] max|gB_out - gB_ref|   = {max_err_gB:.3e}")
    if with_bias:
        print(f"[check] max|gBias_out - gBias_ref| = {max_err_gBias:.3e}")

    # -------- 간단한 assert --------
    # GELU 의 경우 커널 쪽 구현이랑 수식이 조금 다를 수 있어서 여유를 조금 더 줘도 됨.
    if act.lower() == "gelu":
        tol_atol = max(atol, 5e-4)
        tol_rtol = max(rtol, 1e-3)
    else:
        tol_atol = atol
        tol_rtol = rtol

    def _assert_close(name, out, ref, max_err):
        if ref is None and out is None:
            return
        if ref is None and out is not None:
            raise AssertionError(f"{name}: ref is None but out is not None")
        ok = np.allclose(out, ref, atol=tol_atol, rtol=tol_rtol)
        status = "OK" if ok else "FAIL"
        print(f"[assert] {name}: {status} (max_err={max_err:.3e}, atol={tol_atol}, rtol={tol_rtol})")
        if not ok:
            raise AssertionError(f"{name} mismatch")

    _assert_close("Y",     Y_out,    Y_ref,    max_err_Y)
    _assert_close("Z",     Z_out,    Z_ref,    max_err_Z)
    _assert_close("gA",    gA_out,   gA_ref,   max_err_gA)
    _assert_close("gB",    gB_out,   gB_ref,   max_err_gB)
    if with_bias:
        _assert_close("gBias", gBias_out, gBias_ref, max_err_gBias)

    print("[smoke-raw] GEMM binding forward_raw/backward_raw correctness OK.\n")


if __name__ == "__main__":
    # 기본 케이스 몇 개 돌려보기
    run_case_raw(with_bias=True,  act="relu")
    run_case_raw(with_bias=False, act="none")

    # 필요하면 다른 활성화도 추가로 검증 가능
    # run_case_raw(with_bias=True,  act="leakyrelu")
    # run_case_raw(with_bias=True,  act="sigmoid")
    # run_case_raw(with_bias=True,  act="tanh")
    # run_case_raw(with_bias=True,  act="gelu")

    print("[smoke] all done.")
