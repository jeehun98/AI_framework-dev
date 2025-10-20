# graph_executor_v2/python/test/layers/test_conv2d_layer_standalone.py
import os, sys
import cupy as cp

# --- add project root to sys.path ---
THIS = os.path.abspath(os.path.dirname(__file__))                 # .../python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))            # .../graph_executor_v2 (pkg root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.conv2d import Conv2D


def stats(x, name):
    print(f"{name}: shape={tuple(x.shape)}, max={float(cp.max(cp.abs(x))):.3e}, "
          f"norm={float(cp.linalg.norm(x).astype(cp.float32)):.3e}")


def run_basic():
    print("\n[Case 1] Basic forward/backward (act=none)")
    cp.random.seed(1)
    x = cp.random.randn(2, 3, 32, 32).astype(cp.float32)

    conv = Conv2D(kernel_size=(3,3), stride=(1,1), padding=(1,1), use_bias=True)
    y = conv(x)
    print(f"Forward OK: y.shape={y.shape}")
    assert cp.allclose(y, conv.last_z, atol=1e-6)
    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = conv.backward(gy)
    print("Backward OK.")
    stats(dx, "dx"); stats(conv.dW, "dW"); stats(conv.db, "db")


def run_groups():
    print("\n[Case 2] groups=2 check")
    cp.random.seed(2)
    x = cp.random.randn(2, 4, 16, 16).astype(cp.float32)
    conv = Conv2D(filters=8, kernel_size=(3,3), padding=(1,1), groups=2, use_bias=True)
    y = conv(x)
    print(f"Forward OK: y.shape={y.shape}")
    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = conv.backward(gy)
    stats(dx, "dx"); stats(conv.dW, "dW")


def run_stride_pad_dil():
    print("\n[Case 3] stride/pad/dilation variations")
    cp.random.seed(3)
    x = cp.random.randn(1, 3, 17, 19).astype(cp.float32)
    conv = Conv2D(filters=5, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1), use_bias=False)
    y = conv(x)
    print(f"Forward OK: y.shape={y.shape}")
    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = conv.backward(gy)
    stats(dx, "dx")
    print(f"Output shape matches computed: {y.shape == conv.compute_output_shape(x.shape)}")


def run_repeat():
    print("\n[Case 4] workspace reuse check")
    cp.random.seed(4)
    x = cp.random.randn(2, 3, 24, 24).astype(cp.float32)
    conv = Conv2D(filters=6, kernel_size=(3,3), padding=(1,1), use_bias=True)
    for i in range(2):
        y = conv(x)
        gy = cp.random.randn(*y.shape).astype(cp.float32)
        dx = conv.backward(gy)
        print(f"Pass {i+1}: y.shape={y.shape}, dx.shape={dx.shape}")
        stats(conv.dW, f"dW pass{i+1}")
        stats(conv.db, f"db pass{i+1}")


def run_fd():
    print("\n[Case 5] finite-diff gradient sanity (tiny tensor)")
    cp.random.seed(5)
    x = cp.random.randn(1, 1, 7, 7).astype(cp.float32)
    conv = Conv2D(filters=1, kernel_size=(3,3), padding=(1,1), use_bias=True)
    y = conv(x)
    gy = cp.ones_like(y)
    conv.backward(gy)
    dw_analytic = float(conv.dW[0,0,0,0])

    eps = 1e-3
    W0 = conv.W.copy()
    conv.W[0,0,0,0] = W0[0,0,0,0] + eps
    y_pos = conv(x)
    loss_pos = float(cp.sum(y_pos))
    conv.W[0,0,0,0] = W0[0,0,0,0] - eps
    y_neg = conv(x)
    loss_neg = float(cp.sum(y_neg))
    conv.W[...] = W0
    dw_num = (loss_pos - loss_neg) / (2*eps)
    rel_err = abs(dw_num - dw_analytic) / (abs(dw_num) + 1e-6)
    print(f"dW[0,0,0,0]: analytic={dw_analytic:.4f}, numeric={dw_num:.4f}, rel.err={rel_err:.4e}")
    assert rel_err < 1e-1


if __name__ == "__main__":
    print("=== Conv2D Layer Standalone Test ===")
    run_basic()
    run_groups()
    run_stride_pad_dil()
    run_repeat()
    run_fd()
    print("\n[All Conv2D layer tests OK ✅]")
