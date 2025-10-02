# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))                      # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))           # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------


import cupy as cp
from graph_executor_v2.layers.conv2d import Conv2D

def main():
    cp.random.seed(1)
    x = cp.random.randn(8, 3, 32, 32).astype(cp.float32)  # NCHW

    conv = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),   # same-ish
        dilation=(1, 1),
        groups=1,
        use_bias=True,
        initializer="he",
    )
    y = conv(x)
    print(f"[Conv2D] forward OK: {y.shape}")

    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = conv.backward(gy)  # 바인딩 backward 필요
    assert dx.shape == x.shape, f"dx.shape={dx.shape}"
    assert conv.dW is not None
    if conv.use_bias:
        assert conv.db is not None
    print("[Conv2D] backward OK")

if __name__ == "__main__":
    try:
        main()
        print("[Conv2D] all good ✅")
    except NotImplementedError as e:
        print(f"[Conv2D] backward not implemented in binding → SKIP: {e}")
