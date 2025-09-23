import os, sys, numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
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

from graph_executor_v2 import _core as ge

# === 테스트 데이터 ===
B,H,M,N,D = 2, 3, 4, 5, 8
Q = np.random.randn(B,H,M,D).astype(np.float32)
K = np.random.randn(B,H,N,D).astype(np.float32)
V = np.random.randn(B,H,N,D).astype(np.float32)

# === 레퍼런스 계산 (numpy) ===
s = 1.0 / np.sqrt(D)
S = np.einsum("bhmd,bhnd->bhmn", Q, K) * s
P = np.exp(S - S.max(axis=-1, keepdims=True))
P /= P.sum(axis=-1, keepdims=True)
Y_ref = np.einsum("bhmn,bhnd->bhmd", P, V)

# === 프레임워크 호출 ===
Y = ge.sdpa(Q, K, V)  # scale=auto, mask/dropout off

def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "cpu") and callable(x.cpu):
        return np.asarray(x.cpu())
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)

Y_np = to_numpy(Y)
print("Y type:", type(Y), "→", type(Y_np), getattr(Y_np, "dtype", None), getattr(Y_np, "shape", None))
print("Y_ref type:", type(Y_ref), getattr(Y_ref, "dtype", None), getattr(Y_ref, "shape", None))

if Y_np is None:
    print("❌ ge.sdpa returned None")
else:
    print("allclose:", np.allclose(Y_np, Y_ref.astype(np.float32), atol=1e-4))
