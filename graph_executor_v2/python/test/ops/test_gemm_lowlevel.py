# python/test/ops/test_gemm_lowlevel.py
# ============================================================
# Low-level GEMM bindings tests (ai::Tensor 기반)
# - Backend: CuPy -> Torch fallback (CUDA 필수)
# - Targets:
#   * forward/backward NN path (row-major, f32)
#   * save_z 경로 동작 확인
#   * gBias PerN 강제 가드 (PerM 형태는 에러)
#   * 다양한 activation(epilogue) 점검
#   * (선택) numpy helper와 수치 비교
# ============================================================

import os, sys
import numpy as np
import pytest

# --- import path / CUDA DLLs (Windows) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

# --- pick GPU array backend: CuPy -> Torch fallback ---
xp = None
cupy = None
torch = None
use_cupy = False
use_torch = False
_backend_name = None
try:
    import cupy as cp
    _ = cp.arange(1, dtype=cp.float32)  # sanity
    cupy = cp
    xp = cp
    use_cupy = True
    _backend_name = "cupy"
except Exception:
    try:
        import torch as _torch
        assert _torch.cuda.is_available(), "CUDA not available for torch"
        torch = _torch
        xp = _torch
        use_torch = True
        _backend_name = "torch"
    except Exception as e:
        raise RuntimeError("Need CuPy or PyTorch CUDA to run low-level tests.") from e

# --- load ops ---
from graph_executor_v2.ops import require
ops_gemm = require("gemm")  # graph_executor_v2/ops/_ops_gemm.pyd


# =========================
# helpers
# =========================
def check_no_debug_string(pyd_path: str, needle=b"[BWD dbg]"):
    try:
        with open(pyd_path, "rb") as f:
            return (needle not in f.read())
    except Exception:
        return False

def make_ptr(arr):
    """Return device pointer (int) from CuPy or Torch tensor."""
    if use_cupy:
        return int(arr.data.ptr)
    else:
        return arr.data_ptr()

def to_numpy(arr):
    if use_cupy:
        return cupy.asnumpy(arr)
    else:
        return arr.detach().cpu().numpy()

def randn(shape, dtype="float32", seed=None):
    if seed is not None:
        if use_cupy:
            cupy.random.seed(seed)
        else:
            torch.manual_seed(seed)
    if use_cupy:
        return cupy.random.standard_normal(shape, dtype=cupy.float32 if dtype=="float32" else None)
    else:
        return torch.randn(*shape, device="cuda", dtype=torch.float32)

def empty(shape, dtype="float32"):
    if use_cupy:
        return cupy.empty(shape, dtype=cupy.float32 if dtype=="float32" else None)
    else:
        return torch.empty(*shape, device="cuda", dtype=torch.float32)

def synchronize():
    if use_cupy:
        cupy.cuda.Stream.null.synchronize()
    else:
        torch.cuda.synchronize()

def reshape_1xN(x):
    return x.reshape(1, -1) if use_cupy else x.view(1, -1)

def act_to_enum(act: str):
    """Map string to ops_gemm.ActKind enum."""
    return getattr(
        ops_gemm.ActKind,
        "ReLU"     if act=="relu" else
        "LeakyReLU" if act in ("leakyrelu","leaky_relu","lrelu") else
        "GELU"     if act=="gelu" else
        "Sigmoid"  if act=="sigmoid" else
        "Tanh"     if act=="tanh" else
        "None"
    )

# =========================
# fixtures
# =========================
@pytest.fixture(scope="session")
def backend_name():
    return _backend_name

@pytest.fixture(scope="session")
def binary_path():
    return ops_gemm.__file__

@pytest.fixture(scope="session")
def has_common_types():
    try:
        from graph_executor_v2.ops import _ops_common as common
        assert ops_gemm.ActKind is common.ActKind
        assert ops_gemm.GemmAttrs is common.GemmAttrs
        assert ops_gemm.Tensor is common.Tensor
        assert ops_gemm.make_tensor_2d is common.make_tensor_2d
        return True
    except Exception:
        return False

# =========================
# smoke / meta
# =========================
def test_meta_smoke(backend_name, binary_path, has_common_types):
    print("BACKEND:", backend_name)
    print("BINARY :", binary_path)
    assert os.path.isfile(binary_path)
    assert has_common_types
    assert check_no_debug_string(binary_path), "Binary should not contain debug needle"

# =========================
# core low-level forward/backward
# =========================
@pytest.mark.parametrize("M,K,N", [
    (8, 16, 12),
    (32, 64, 48),
    (64, 128, 32),
])
@pytest.mark.parametrize("act,leaky", [
    ("none", 0.01),
    ("relu", 0.01),
    ("leakyrelu", 0.2),
    ("tanh", 0.01),
    ("sigmoid", 0.01),
    ("gelu", 0.01),
])
def test_forward_backward_lowlevel(M, K, N, act, leaky):
    # inputs
    A = randn((M, K), seed=123)
    B = randn((K, N), seed=321)
    Bias = randn((N,), seed=777)  # PerN
    Y = empty((M, N))
    gY = randn((M, N), seed=999)

    # Precompute Z = A@B + bias
    if use_cupy:
        Z = A @ B + Bias.reshape(1, N)
    else:
        Z = A @ B + Bias.view(1, N)

    # wrap as ai::Tensor (zero-copy)
    A_t = ops_gemm.make_tensor_2d(make_ptr(A), [M, K])
    B_t = ops_gemm.make_tensor_2d(make_ptr(B), [K, N])
    Bias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(Bias)), [1, N])
    Y_t = ops_gemm.make_tensor_2d(make_ptr(Y), [M, N])
    gY_t = ops_gemm.make_tensor_2d(make_ptr(gY), [M, N])
    Z_t = ops_gemm.make_tensor_2d(make_ptr(Z), [M, N])

    gA = empty((M, K));  gA_t = ops_gemm.make_tensor_2d(make_ptr(gA), [M, K])
    gB = empty((K, N));  gB_t = ops_gemm.make_tensor_2d(make_ptr(gB), [K, N])
    gBias = empty((N,)); gBias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(gBias)), [1, N])

    # attrs
    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = act_to_enum(act)
    attrs.with_bias = True
    attrs.leaky_slope = float(leaky)
    attrs.save_z = False  # 이 케이스는 외부에서 Z 미리 계산

    # forward
    ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None, None)
    synchronize()
    assert Y.shape == (M, N)

    # backward
    ops_gemm.backward(A_t, B_t, None, gY_t, Z_t, gA_t, gB_t, None, gBias_t, attrs, None)
    synchronize()
    assert gA.shape == (M, K) and gB.shape == (K, N) and gBias.shape == (N,)

# =========================
# save_z path (Z_saved를 forward에서 저장해 두고 backward에 사용)
# =========================
def test_forward_with_save_z_then_backward():
    M, K, N = 16, 32, 24
    A = randn((M, K), seed=1234)
    B = randn((K, N), seed=4321)
    Bias = randn((N,), seed=42)
    Y = empty((M, N))
    Z_saved = empty((M, N))
    gY = randn((M, N), seed=2025)

    A_t = ops_gemm.make_tensor_2d(make_ptr(A), [M, K])
    B_t = ops_gemm.make_tensor_2d(make_ptr(B), [K, N])
    Bias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(Bias)), [1, N])
    Y_t = ops_gemm.make_tensor_2d(make_ptr(Y), [M, N])
    Z_t = ops_gemm.make_tensor_2d(make_ptr(Z_saved), [M, N])  # forward에서 채워질 예정
    gY_t = ops_gemm.make_tensor_2d(make_ptr(gY), [M, N])

    gA = empty((M, K));  gA_t = ops_gemm.make_tensor_2d(make_ptr(gA), [M, K])
    gB = empty((K, N));  gB_t = ops_gemm.make_tensor_2d(make_ptr(gB), [K, N])
    gBias = empty((N,)); gBias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(gBias)), [1, N])

    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = act_to_enum("relu")
    attrs.with_bias = True
    attrs.leaky_slope = 0.01
    attrs.save_z = True  # 명시적으로 저장

    # forward(save_z)
    ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, Z_t, None)
    synchronize()

    # backward(Z_saved 사용)
    ops_gemm.backward(A_t, B_t, None, gY_t, Z_t, gA_t, gB_t, None, gBias_t, attrs, None)
    synchronize()

    # 기본 형태 검사
    assert Y.shape == (M, N)
    assert Z_saved.shape == (M, N)

# =========================
# gBias PerN guard (PerM 형태를 주면 에러)
# =========================
def test_gbias_perm_guard_raises():
    M, K, N = 8, 16, 12
    A = randn((M, K), seed=1)
    B = randn((K, N), seed=2)
    Bias = randn((N,), seed=3)
    Y = empty((M, N))
    gY = randn((M, N), seed=4)

    # Z = A@B + bias
    if use_cupy:
        Z = A @ B + Bias.reshape(1, N)
    else:
        Z = A @ B + Bias.view(1, N)

    A_t = ops_gemm.make_tensor_2d(make_ptr(A), [M, K])
    B_t = ops_gemm.make_tensor_2d(make_ptr(B), [K, N])
    Bias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(Bias)), [1, N])
    Y_t = ops_gemm.make_tensor_2d(make_ptr(Y), [M, N])
    gY_t = ops_gemm.make_tensor_2d(make_ptr(gY), [M, N])
    Z_t = ops_gemm.make_tensor_2d(make_ptr(Z), [M, N])

    # 잘못된 형태의 gBias(PerM) 준비: (M,) 또는 (M,1)
    gBias_bad = empty((M,))
    gBias_bad_t = ops_gemm.make_tensor_2d(make_ptr(gBias_bad.reshape(M, 1) if use_cupy else gBias_bad.view(M,1)), [M, 1])

    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = act_to_enum("none")
    attrs.with_bias = True
    attrs.leaky_slope = 0.01

    # forward OK
    ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None, None)
    synchronize()

    # backward에서 에러 기대
    with pytest.raises(Exception) as ei:
        ops_gemm.backward(A_t, B_t, None, gY_t, Z_t, None, None, None, gBias_bad_t, attrs, None)
    msg = str(ei.value).lower()
    assert "pern" in msg or "perm" in msg, f"unexpected error message: {msg}"

# =========================
# (선택) numpy helper와 값 비교
# =========================
@pytest.mark.parametrize("act,atol,rtol", [
    ("none",    1e-3, 1e-3),
    ("relu",    1e-3, 1e-3),
    ("leakyrelu", 1e-3, 1e-3),
    ("tanh",    3e-3, 3e-3),
    ("sigmoid", 3e-3, 3e-3),
    ("gelu",    3e-3, 3e-3),
])
def test_compare_with_numpy_helper(act, atol, rtol):
    # 작은 케이스로 수치 비교
    M, K, N = 8, 16, 12
    A = randn((M, K), seed=11); B = randn((K, N), seed=22); Bias = randn((N,), seed=33)
    Y = empty((M, N)); gY = randn((M, N), seed=44)

    # Z for backward
    if use_cupy:
        Z = A @ B + Bias.reshape(1, N)
    else:
        Z = A @ B + Bias.view(1, N)

    # wrap
    A_t = ops_gemm.make_tensor_2d(make_ptr(A), [M, K])
    B_t = ops_gemm.make_tensor_2d(make_ptr(B), [K, N])
    Bias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(Bias)), [1, N])
    Y_t = ops_gemm.make_tensor_2d(make_ptr(Y), [M, N])
    gY_t = ops_gemm.make_tensor_2d(make_ptr(gY), [M, N])
    Z_t = ops_gemm.make_tensor_2d(make_ptr(Z), [M, N])

    gA = empty((M, K));  gA_t = ops_gemm.make_tensor_2d(make_ptr(gA), [M, K])
    gB = empty((K, N));  gB_t = ops_gemm.make_tensor_2d(make_ptr(gB), [K, N])
    gBias = empty((N,)); gBias_t = ops_gemm.make_tensor_2d(make_ptr(reshape_1xN(gBias)), [1, N])

    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = act_to_enum(act)
    attrs.with_bias = True
    attrs.leaky_slope = 0.2 if act in ("leakyrelu",) else 0.01  # leaky는 slope 조금 키워도 비교 OK

    # forward/backward (low-level)
    ops_gemm.forward(A_t, B_t, Bias_t, Y_t, attrs, None, None)
    synchronize()
    ops_gemm.backward(A_t, B_t, None, gY_t, Z_t, gA_t, gB_t, None, gBias_t, attrs, None)
    synchronize()

    # numpy helper (host)
    A_np, B_np, Bias_np, gY_np, Z_np = map(to_numpy, (A, B, Bias, gY, Z))
    Y_ref = ops_gemm.forward_numpy(A_np, B_np, Bias_np, act=act, leaky_slope=float(attrs.leaky_slope))
    out   = ops_gemm.backward_numpy(A_np, B_np, gY_np, Z_np,
                                    act=act, bias_kind="pern", leaky_slope=float(attrs.leaky_slope))
    gA_ref, gB_ref, gBias_ref = out["gA"], out["gB"], out["gBias"]

    def _allclose(x, y, name):
        ok = np.allclose(x, y, atol=atol, rtol=rtol)
        if not ok:
            diff = np.max(np.abs(x - y))
            print(f"[{name}] max abs diff: {diff} (atol={atol}, rtol={rtol})")
        assert ok, f"{name} mismatch"

    _allclose(to_numpy(Y), np.asarray(Y_ref), "Y")
    _allclose(to_numpy(gA), gA_ref, "gA")
    _allclose(to_numpy(gB), gB_ref, "gB")
    if gBias_ref is not None:
        _allclose(to_numpy(gBias), gBias_ref, "gBias")


# ============================================================
# Optional: quick CLI run without pytest
#   python -m pytest -q python/test/ops/test_gemm_lowlevel.py
# or:
#   python python/test/ops/test_gemm_lowlevel.py
# ============================================================
if __name__ == "__main__":
    # 간단 스모크만 실행
    test_forward_backward_lowlevel(8,16,12,"relu",0.01)
    print("Smoke OK with backend:", _backend_name)
