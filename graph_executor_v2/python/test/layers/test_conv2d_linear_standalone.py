# python/test/layers/test_conv2d_linear_standalone.py
import os, sys
import cupy as cp

# sys.path
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.linear import Linear

def stats(x, name):
    print(f"{name}: shape={tuple(x.shape)}, max={float(cp.max(cp.abs(x))):.3e}, "
          f"norm={float(cp.linalg.norm(x).astype(cp.float32)):.3e}")

if __name__ == "__main__":
    print("=== Conv2D → Flatten → Linear (GEMM) Standalone ===")
    cp.random.seed(0)

    # Input
    x = cp.random.randn(4, 3, 16, 16).astype(cp.float32)  # NCHW

    # Layers
    conv = Conv2D(filters=8, kernel_size=(3,3), padding=(1,1), stride=(1,1), use_bias=True, initializer="he")
    flat = Flatten()
    # build을 프레임워크가 자동 호출하지 않는다면 수동으로 한 번 호출(선택)
    conv.build(x.shape)
    y1 = conv(x)

    z1 = conv.last_z  # pre-activation (act='none' → Y==Z)
    stats(y1, "conv.y")

    f1 = flat(y1)
    stats(f1, "flatten.out")

    lin = Linear(out_features=10, use_bias=True, initializer="xavier")
    lin.build(f1.shape)
    y2 = lin(f1)
    stats(y2, "linear.y")

    # --- Loss: sum(y2) → gY = ones
    gy2 = cp.ones_like(y2, dtype=cp.float32)
    # Backprop
    g_f1 = lin.backward(gy2)
    g_y1 = flat.backward(g_f1)
    g_x  = conv.backward(g_y1)

    print("\n--- Grad stats ---")
    stats(g_x, "grad x")
    stats(lin.dW, "grad W_linear")
    stats(conv.dW, "grad W_conv")
    if lin.db is not None: stats(lin.db, "grad b_linear")
    if conv.db is not None: stats(conv.db, "grad b_conv")

    print("\n[OK] End-to-end Conv2D + GEMM works ✅")
