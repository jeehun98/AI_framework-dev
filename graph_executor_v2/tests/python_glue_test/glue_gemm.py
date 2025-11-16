import os
import sys
from typing import Optional, Tuple

# repo 루트: .../graph_executor_v2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# python/ 디렉토리를 sys.path에 추가
PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import numpy as np
import cupy as cp

from graph_executor_v2.ops import gemm as gemm_ops


# ============================================================
#  Activation (fwd/bwd) 레퍼런스 구현 (NumPy)
# ============================================================

def act_forward_np(z: np.ndarray, act: str, leaky_slope: float = 0.01) -> np.ndarray:
    act = (act or "none").lower()
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
        from math import sqrt
        return 0.5 * z * (1.0 + np.erf(z / np.float32(sqrt(2.0))))
    raise ValueError(f"unsupported act: {act}")


def act_backward_np(z: np.ndarray,
                    gy: np.ndarray,
                    act: str,
                    leaky_slope: float = 0.01) -> np.ndarray:
    """gZ = dL/dZ = dL/dY * dY/dZ"""
    act = (act or "none").lower()
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
                     bias_h: Optional[np.ndarray],
                     act: str,
                     leaky_slope: float = 0.01):
    """
    Z = A @ B + bias
    Y = act(Z)
    bias: None or shape (1, N)
    """
    Z = A_h @ B_h  # (M,K) @ (K,N) -> (M,N)
    if bias_h is not None:
        Z = Z + bias_h.astype(A_h.dtype)
    Y = act_forward_np(Z, act, leaky_slope)
    return Y, Z


def gemm_backward_ref(A_h: np.ndarray,
                      B_h: np.ndarray,
                      bias_h: Optional[np.ndarray],
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
#  glue 레벨: alloc 경로 테스트 (forward/backward)
# ============================================================

def run_case_glue_alloc(m=32, k=64, n=16,
                        with_bias=True,
                        act="relu",
                        leaky_slope=0.01,
                        atol=1e-5, rtol=1e-4):
    print(f"[smoke-glue-alloc] m={m}, k={k}, n={n}, with_bias={with_bias}, act={act}")

    rng = np.random.default_rng(2025)

    # host 데이터
    A_h = rng.standard_normal((m, k), dtype=np.float32)
    B_h = rng.standard_normal((k, n), dtype=np.float32)
    gY_h = rng.standard_normal((m, n), dtype=np.float32)

    if with_bias:
        bias_h = rng.standard_normal((1, n), dtype=np.float32)
    else:
        bias_h = None

    # 레퍼런스
    Y_ref, Z_ref = gemm_forward_ref(A_h, B_h, bias_h, act, leaky_slope)
    gA_ref, gB_ref, gBias_ref = gemm_backward_ref(A_h, B_h, bias_h, gY_h, Z_ref, act, leaky_slope)

    # GPU 버퍼
    A_d = cp.asarray(A_h)
    B_d = cp.asarray(B_h)
    gY_d = cp.asarray(gY_h)
    bias_d = cp.asarray(bias_h) if bias_h is not None else None

    print("[smoke-glue-alloc] running gemm_ops.forward() ...")
    Y_d, Z_d = gemm_ops.forward(
        A_d,
        B_d,
        bias=bias_d,
        act=act,
        with_bias=with_bias,
        leaky_slope=leaky_slope,
        save_z=True,
        return_z=True,
    )

    print("[smoke-glue-alloc] running gemm_ops.backward() ...")
    outs = gemm_ops.backward(
        A_d,
        B_d,
        gY_d,
        Z_d,
        act=act,
        with_bias=with_bias,
        leaky_slope=leaky_slope,
        C=None,
        want_gA=True,
        want_gB=True,
        want_gBias=with_bias,
    )

    gA_d = outs.get("gA", None)
    gB_d = outs.get("gB", None)
    gBias_d = outs.get("gBias", None)

    # 결과 → host
    Y_out = cp.asnumpy(Y_d)
    Z_out = cp.asnumpy(Z_d)
    gA_out = cp.asnumpy(gA_d) if gA_d is not None else None
    gB_out = cp.asnumpy(gB_d) if gB_d is not None else None
    gBias_out = cp.asnumpy(gBias_d) if gBias_d is not None else None

    # diff
    max_err_Y = float(np.max(np.abs(Y_out - Y_ref)))
    max_err_Z = float(np.max(np.abs(Z_out - Z_ref)))
    max_err_gA = float(np.max(np.abs(gA_out - gA_ref)))
    max_err_gB = float(np.max(np.abs(gB_out - gB_ref)))
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

    _assert_close("Y", Y_out, Y_ref, max_err_Y)
    _assert_close("Z", Z_out, Z_ref, max_err_Z)
    _assert_close("gA", gA_out, gA_ref, max_err_gA)
    _assert_close("gB", gB_out, gB_ref, max_err_gB)
    if with_bias:
        _assert_close("gBias", gBias_out, gBias_ref, max_err_gBias)

    print("[smoke-glue-alloc] glue forward/backward correctness OK.\n")


# ============================================================
#  glue 레벨: capture-safe 경로 테스트 (forward_into/backward_into)
# ============================================================

def run_case_glue_capture(m=32, k=64, n=16,
                          with_bias=True,
                          act="relu",
                          leaky_slope=0.01,
                          atol=1e-5, rtol=1e-4):
    print(f"[smoke-glue-capture] m={m}, k={k}, n={n}, with_bias={with_bias}, act={act}")

    rng = np.random.default_rng(2025)

    # host 데이터
    A_h = rng.standard_normal((m, k), dtype=np.float32)
    B_h = rng.standard_normal((k, n), dtype=np.float32)
    gY_h = rng.standard_normal((m, n), dtype=np.float32)
    if with_bias:
        bias_h = rng.standard_normal((1, n), dtype=np.float32)
    else:
        bias_h = None

    # 레퍼런스
    Y_ref, Z_ref = gemm_forward_ref(A_h, B_h, bias_h, act, leaky_slope)
    gA_ref, gB_ref, gBias_ref = gemm_backward_ref(A_h, B_h, bias_h, gY_h, Z_ref, act, leaky_slope)

    # GPU 버퍼
    A_d = cp.asarray(A_h)
    B_d = cp.asarray(B_h)
    gY_d = cp.asarray(gY_h)
    bias_d = cp.asarray(bias_h) if bias_h is not None else None

    Y_d = cp.empty((m, n), dtype=cp.float32)
    Z_d = cp.empty((m, n), dtype=cp.float32)
    gA_d = cp.empty((m, k), dtype=cp.float32)
    gB_d = cp.empty((k, n), dtype=cp.float32)
    gBias_d = cp.empty((1, n), dtype=cp.float32) if with_bias else None
    work_dZ = cp.empty((m, n), dtype=cp.float32)

    # lt_workspace는 이번 스모크에선 사용 안 함 (0 bytes)
    lt_workspace = None

    # capture-safe forward_into
    print("[smoke-glue-capture] running gemm_ops.forward_into() ...")
    gemm_ops.forward_into(
        A_d,
        B_d,
        out=Y_d,
        bias=bias_d,
        act=act,
        with_bias=with_bias,
        leaky_slope=leaky_slope,
        save_z=True,
        z_out=Z_d,
        stream=None,
    )

    # capture-safe backward_into
    print("[smoke-glue-capture] running gemm_ops.backward_into() ...")
    gemm_ops.backward_into(
        A_d,
        B_d,
        gY_d,
        Z_d,
        act=act,
        with_bias=with_bias,
        leaky_slope=leaky_slope,
        C=None,
        gA_out=gA_d,
        gB_out=gB_d,
        gC_out=None,
        gBias_out=gBias_d,
        stream=None,
        work_dZ=work_dZ,
        lt_workspace=lt_workspace,
    )

    # 결과 → host
    Y_out = cp.asnumpy(Y_d)
    Z_out = cp.asnumpy(Z_d)
    gA_out = cp.asnumpy(gA_d)
    gB_out = cp.asnumpy(gB_d)
    gBias_out = cp.asnumpy(gBias_d) if gBias_d is not None else None

    # diff
    max_err_Y = float(np.max(np.abs(Y_out - Y_ref)))
    max_err_Z = float(np.max(np.abs(Z_out - Z_ref)))
    max_err_gA = float(np.max(np.abs(gA_out - gA_ref)))
    max_err_gB = float(np.max(np.abs(gB_out - gB_ref)))
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

    _assert_close("Y", Y_out, Y_ref, max_err_Y)
    _assert_close("Z", Z_out, Z_ref, max_err_Z)
    _assert_close("gA", gA_out, gA_ref, max_err_gA)
    _assert_close("gB", gB_out, gB_ref, max_err_gB)
    if with_bias:
        _assert_close("gBias", gBias_out, gBias_ref, max_err_gBias)

    print("[smoke-glue-capture] glue forward_into/backward_into correctness OK.\n")


if __name__ == "__main__":
    # alloc 경로
    run_case_glue_alloc(with_bias=True, act="relu")
    run_case_glue_alloc(with_bias=False, act="none")

    # capture-safe 경로
    run_case_glue_capture(with_bias=True, act="relu")
    run_case_glue_capture(with_bias=False, act="none")

    print("[smoke-glue] all done.")
