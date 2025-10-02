# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))                      # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))           # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp
from graph_executor_v2.layers.dense_gemm import Dense

def run_once(use_native_bwd: bool):
    cp.random.seed(0)
    x = cp.random.randn(32, 64).astype(cp.float32)

    layer = Dense(
        units=128,
        activation="relu",
        initializer="he",
        use_native_bwd=use_native_bwd
    )
    y = layer(x)
    assert y.shape == (32, 128), f"y.shape={y.shape}"
    print(f"[Dense] forward OK: {y.shape}")

    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = layer.backward(gy)
    assert dx.shape == x.shape, f"dx.shape={dx.shape}"
    assert layer.dW is not None and layer.db is not None
    print(f"[Dense] backward OK (use_native_bwd={use_native_bwd})")

if __name__ == "__main__":
    run_once(use_native_bwd=False)
    run_once(use_native_bwd=True)
    print("[Dense] all good ✅")
