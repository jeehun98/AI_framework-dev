import os
import sys
from typing import Optional, Tuple

# repo 루트: .../graph_executor_v2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import numpy as np
import cupy as cp

from graph_executor_v2.layers.dense_gemm import Dense


# ============================================================
#  Activation + Dense 레퍼런스 구현 (NumPy)
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


def dense_forward_ref(x: np.ndarray,
                      W: np.ndarray,
                      b: np.ndarray,
                      act: str,
                      leaky_slope: float = 0.01):
    """
    x: (M, in_dim)
    W: (in_dim, units)
    b: (1, units)
    """
    z = x @ W + b       # (M,units) + (1,units)
    y = act_forward_np(z, act, leaky_slope)
    return y, z


def dense_backward_ref(x: np.ndarray,
                       W: np.ndarray,
                       b: np.ndarray,
                       gY: np.ndarray,
                       Z: np.ndarray,
                       act: str,
                       leaky_slope: float = 0.01):
    """
    gZ = gY * act'(Z)
    gW = x^T @ gZ
    gB = sum(gZ, axis=0, keepdims=True)
    dx = gZ @ W^T
    """
    gZ = act_backward_np(Z, gY, act, leaky_slope)
    gW = x.T @ gZ
    gB = gZ.sum(axis=0, keepdims=True)
    dx = gZ @ W.T
    return dx, gW, gB


# ============================================================
#  레이어 레벨: call/backward (use_native_bwd={False,True})
# ============================================================

def run_case_dense_layer(m=32, in_dim=64, units=16,
                         act="relu",
                         leaky_slope=0.01,
                         use_native_bwd=False,
                         atol=1e-5, rtol=1e-4):
    mode = "native" if use_native_bwd else "manual"
    print(f"[smoke-dense-{mode}] m={m}, in_dim={in_dim}, units={units}, act={act}")

    rng = np.random.default_rng(2025)

    # host 데이터
    x_h = rng.standard_normal((m, in_dim), dtype=np.float32)
    W_h = rng.standard_normal((in_dim, units), dtype=np.float32)
    b_h = rng.standard_normal((1, units), dtype=np.float32)
    gY_h = rng.standard_normal((m, units), dtype=np.float32)

    # 레퍼런스
    y_ref, z_ref = dense_forward_ref(x_h, W_h, b_h, act, leaky_slope)
    dx_ref, gW_ref, gB_ref = dense_backward_ref(x_h, W_h, b_h, gY_h, z_ref, act, leaky_slope)

    # GPU 버퍼
    x_d = cp.asarray(x_h)
    W_d = cp.asarray(W_h)
    b_d = cp.asarray(b_h)
    gY_d = cp.asarray(gY_h)

    # 레이어 생성
    dense = Dense(
        units=units,
        activation=act,
        initializer="zeros",  # build 에서 초기화는 무시하고 아래에서 W,b 직접 덮어씀
        use_native_bwd=use_native_bwd,
        name="dense_test",
    )
    dense.build((m, in_dim))

    # 파라미터 직접 세팅
    dense.W[...] = W_d
    dense.b[...] = b_d
    dense.zero_grad()

    # forward
    print(f"[smoke-dense-{mode}] running Dense.call() ...")
    y_out_d = dense.call(x_d)
    y_out = cp.asnumpy(y_out_d)

    # backward
    print(f"[smoke-dense-{mode}] running Dense.backward() ...")
    dx_d = dense.backward(gY_d)
    dx_out = cp.asnumpy(dx_d)
    gW_out = cp.asnumpy(dense.dW)
    gB_out = cp.asnumpy(dense.db)

    # diff
    max_err_y = float(np.max(np.abs(y_out - y_ref)))
    max_err_dx = float(np.max(np.abs(dx_out - dx_ref)))
    max_err_gW = float(np.max(np.abs(gW_out - gW_ref)))
    max_err_gB = float(np.max(np.abs(gB_out - gB_ref)))

    print(f"[check] max|y_out - y_ref|   = {max_err_y:.3e}")
    print(f"[check] max|dx_out - dx_ref| = {max_err_dx:.3e}")
    print(f"[check] max|gW_out - gW_ref| = {max_err_gW:.3e}")
    print(f"[check] max|gB_out - gB_ref| = {max_err_gB:.3e}")

    if act.lower() == "gelu":
        tol_atol = max(atol, 5e-4)
        tol_rtol = max(rtol, 1e-3)
    else:
        tol_atol = atol
        tol_rtol = rtol

    def _assert_close(name, out, ref, max_err):
        ok = np.allclose(out, ref, atol=tol_atol, rtol=tol_rtol)
        status = "OK" if ok else "FAIL"
        print(f"[assert] {name}: {status} (max_err={max_err:.3e}, atol={tol_atol}, rtol={tol_rtol})")
        if not ok:
            raise AssertionError(f"{name} mismatch")

    _assert_close("y", y_out, y_ref, max_err_y)
    _assert_close("dx", dx_out, dx_ref, max_err_dx)
    _assert_close("gW", gW_out, gW_ref, max_err_gW)
    _assert_close("gB", gB_out, gB_ref, max_err_gB)

    print(f"[smoke-dense-{mode}] Dense.call/backward correctness OK.\n")


# ============================================================
#  레이어 레벨: forward_into/backward_into (capture-safe)
# ============================================================

def run_case_dense_capture(m=32, in_dim=64, units=16,
                           act="relu",
                           leaky_slope=0.01,
                           atol=1e-5, rtol=1e-4):
    print(f"[smoke-dense-capture] m={m}, in_dim={in_dim}, units={units}, act={act}")

    rng = np.random.default_rng(2025)

    # host 데이터
    x_h = rng.standard_normal((m, in_dim), dtype=np.float32)
    W_h = rng.standard_normal((in_dim, units), dtype=np.float32)
    b_h = rng.standard_normal((1, units), dtype=np.float32)
    gY_h = rng.standard_normal((m, units), dtype=np.float32)

    # 레퍼런스
    y_ref, z_ref = dense_forward_ref(x_h, W_h, b_h, act, leaky_slope)
    dx_ref, gW_ref, gB_ref = dense_backward_ref(x_h, W_h, b_h, gY_h, z_ref, act, leaky_slope)

    # GPU 버퍼
    x_d = cp.asarray(x_h)
    W_d = cp.asarray(W_h)
    b_d = cp.asarray(b_h)
    gY_d = cp.asarray(gY_h)

    # 출력/그라드 버퍼 (capture-safe)
    y_out_d = cp.empty((m, units), dtype=cp.float32)
    z_out_d = cp.empty((m, units), dtype=cp.float32)
    gA_out_d = cp.empty((m, in_dim), dtype=cp.float32)
    gW_out_d = cp.empty((in_dim, units), dtype=cp.float32)
    gB_out_d = cp.empty((1, units), dtype=cp.float32)
    work_dZ = cp.empty((m, units), dtype=cp.float32)

    dense = Dense(
        units=units,
        activation=act,
        initializer="zeros",
        use_native_bwd=True,  # capture-safe 경로는 네이티브 backward_into 사용
        name="dense_capture",
    )
    dense.build((m, in_dim))

    dense.W[...] = W_d
    dense.b[...] = b_d
    dense.zero_grad()

    # forward_into
    print("[smoke-dense-capture] running Dense.forward_into() ...")
    dense.forward_into(
        x_d,
        out=y_out_d,
        z_out=z_out_d,
        stream=None,
        work=None,
    )

    # backward_into
    print("[smoke-dense-capture] running Dense.backward_into() ...")
    dense.backward_into(
        grad_output=gY_d,
        gA_out=gA_out_d,
        gW_out=gW_out_d,
        gB_out=gB_out_d,
        work_dZ=work_dZ,
        lt_workspace=None,
        stream=None,
        work=None,
    )

    # 결과 → host
    y_out = cp.asnumpy(y_out_d)
    dx_out = cp.asnumpy(gA_out_d)
    gW_out = cp.asnumpy(gW_out_d)
    gB_out = cp.asnumpy(gB_out_d)

    # diff
    max_err_y = float(np.max(np.abs(y_out - y_ref)))
    max_err_dx = float(np.max(np.abs(dx_out - dx_ref)))
    max_err_gW = float(np.max(np.abs(gW_out - gW_ref)))
    max_err_gB = float(np.max(np.abs(gB_out - gB_ref)))

    print(f"[check] max|y_out - y_ref|   = {max_err_y:.3e}")
    print(f"[check] max|dx_out - dx_ref| = {max_err_dx:.3e}")
    print(f"[check] max|gW_out - gW_ref| = {max_err_gW:.3e}")
    print(f"[check] max|gB_out - gB_ref| = {max_err_gB:.3e}")

    if act.lower() == "gelu":
        tol_atol = max(atol, 5e-4)
        tol_rtol = max(rtol, 1e-3)
    else:
        tol_atol = atol
        tol_rtol = rtol

    def _assert_close(name, out, ref, max_err):
        ok = np.allclose(out, ref, atol=tol_atol, rtol=tol_rtol)
        status = "OK" if ok else "FAIL"
        print(f"[assert] {name}: {status} (max_err={max_err:.3e}, atol={tol_atol}, rtol={tol_rtol})")
        if not ok:
            raise AssertionError(f"{name} mismatch")

    _assert_close("y", y_out, y_ref, max_err_y)
    _assert_close("dx", dx_out, dx_ref, max_err_dx)
    _assert_close("gW", gW_out, gW_ref, max_err_gW)
    _assert_close("gB", gB_out, gB_ref, max_err_gB)

    print("[smoke-dense-capture] Dense.forward_into/backward_into correctness OK.\n")


if __name__ == "__main__":
    # 수동 backward(CuPy) vs NumPy
    run_case_dense_layer(use_native_bwd=False, act="relu")
    run_case_dense_layer(use_native_bwd=False, act="none")

    # 네이티브 backward(gemm_ops.backward) vs NumPy
    run_case_dense_layer(use_native_bwd=True, act="relu")
    run_case_dense_layer(use_native_bwd=True, act="none")

    # capture-safe 경로
    run_case_dense_capture(act="relu")
    run_case_dense_capture(act="none")

    print("[smoke-dense] all done.")
