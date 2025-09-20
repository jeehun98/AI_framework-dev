import os, sys, time, numpy as np
from pathlib import Path

# 1) graph_executor_v2/python 을 import path 에 추가
PKG = str(Path(__file__).resolve().parents[1])
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# 2) 필요한 DLL 폴더들을 미리 등록 (Python 3.8+)
dll_hints = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",  # cudart64_126.dll, cublas*.dll 등
    # 빌드 산출물 폴더에 별도 .dll 이 있다면 여기도 추가
    # r"C:\Users\as042\Desktop\AI_framework-dev\graph_executor_v2\build\Release",
    # VS CRT이 필요할 경우(대개는 불필요하지만 혹시):
    # r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\x64\Microsoft.VC143.CRT",
]
for d in dll_hints:
    if os.path.isdir(d):
        os.add_dll_directory(d)

from graph_executor_v2 import _core as core

def gelu_tanh(x):
    k0 = 0.7978845608
    k1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(k0 * (x + k1 * x**3)))

def apply_act(x, act, leaky=0.01):
    act = act.lower()
    if act in ("none", "identity"):
        return x
    if act == "relu":
        return np.maximum(x, 0.0)
    if act in ("leakyrelu","leaky_relu","lrelu"):
        y = x.copy()
        y[y<0] *= leaky
        return y
    if act == "gelu":
        return gelu_tanh(x)
    if act == "sigmoid":
        return 1.0/(1.0+np.exp(-x))
    if act == "tanh":
        return np.tanh(x)
    raise ValueError(f"unknown act {act}")

def run_case(M,K,N, act="relu", bias_kind="pern", seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M,K), dtype=np.float32)
    B = rng.standard_normal((K,N), dtype=np.float32)

    bias = None
    if bias_kind == "scalar":
        bias = rng.standard_normal((1,), dtype=np.float32)
    elif bias_kind == "perm":
        bias = rng.standard_normal((M,), dtype=np.float32)
    elif bias_kind == "pern":
        bias = rng.standard_normal((N,), dtype=np.float32)
    elif bias_kind == "none":
        bias = None
    else:
        raise ValueError("bias_kind must be one of: none, scalar, perm, pern")

    # Reference
    Z = A @ B
    if bias is not None:
        if bias.shape == (1,):
            Z = Z + bias[0]
        elif bias.shape == (M,):
            Z = Z + bias.reshape(M,1)
        elif bias.shape == (N,):
            Z = Z + bias.reshape(1,N)
    Y_ref = apply_act(Z, act)

    # DUT
    t0 = time.perf_counter()
    Y = core.gemm_bias_act(A, B, bias, act=act)
    t1 = time.perf_counter()

    # Compare
    diff = np.max(np.abs(Y - Y_ref))
    rel = diff / (np.max(np.abs(Y_ref)) + 1e-6)

    print(f"[{M}x{K}]x[{K}x{N}] act={act}, bias={bias_kind}  "
          f"max_abs={diff:.3e} rel={rel:.3e}  time={t1-t0:.4f}s")
    assert Y.shape == (M,N)
    assert Y.dtype == np.float32
    # Allow small numeric error from fast-math; tighten if needed
    assert diff < 5e-4, f"max_abs too large: {diff}"
    return diff, rel

def main():
    cases = [
        (16, 32, 8,  "none",   "none"),
        (32, 64, 16, "relu",   "pern"),
        (64, 64, 64, "gelu",   "scalar"),
        (31, 17, 29, "tanh",   "perm"),
        (128,128,64, "leaky_relu", "pern"),
        (8,  8,  8,  "sigmoid","none"),
    ]
    ok = True
    for (M,K,N,act,bk) in cases:
        try:
            run_case(M,K,N, act=act, bias_kind=bk, seed=42)
        except AssertionError as e:
            print("FAILED:", e)
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()