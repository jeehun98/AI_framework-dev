# graph_executor_v2/python/test/layers/test_mini_cnn_standalone.py
import os, sys
import cupy as cp

# --- add project root to sys.path ---
THIS = os.path.abspath(os.path.dirname(__file__))           # .../python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))      # .../graph_executor_v2 (pkg root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense

# -------------------------------------------------------------------
# Minimal layers implemented locally to avoid extra dependencies
# -------------------------------------------------------------------
class Layer:
    def __init__(self, name=None):
        self.built = True
        self.name = name

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class ReLU(Layer):
    """Y = max(0, X). We store mask from input>0 (pre-activation)."""
    def __init__(self, name=None):
        super().__init__(name=name)
        self.mask = None

    def call(self, x: cp.ndarray) -> cp.ndarray:
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad_output * self.mask

class BatchNorm2D(Layer):
    """
    Training-mode BN for NCHW, per-channel affine.
    Y = gamma * (X - mean)/sqrt(var+eps) + beta
    mean/var over axes (N,H,W).
    """
    def __init__(self, num_channels: int, eps: float=1e-5, name=None):
        super().__init__(name=name)
        self.C = int(num_channels)
        self.eps = float(eps)
        self.gamma = cp.ones((1, self.C, 1, 1), dtype=cp.float32)
        self.beta  = cp.zeros((1, self.C, 1, 1), dtype=cp.float32)
        # cache for backward
        self.x_hat = None
        self.mu = None
        self.var = None
        self.x_centered = None
        self.M = None

    def call(self, x: cp.ndarray) -> cp.ndarray:
        if x.ndim != 4 or x.shape[1] != self.C:
            raise ValueError(f"BatchNorm2D expects NCHW with C={self.C}")
        # per-channel stats over (N,H,W)
        axes = (0, 2, 3)
        self.mu = x.mean(axis=axes, keepdims=True)
        self.var = x.var(axis=axes, keepdims=True)  # by default population variance
        self.x_centered = x - self.mu
        inv_std = cp.reciprocal(cp.sqrt(self.var + self.eps))
        self.x_hat = self.x_centered * inv_std
        self.M = x.size // self.C  # per-channel count
        y = self.gamma * self.x_hat + self.beta
        return y

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if any(v is None for v in (self.x_hat, self.mu, self.var, self.x_centered, self.M)):
            raise RuntimeError("BatchNorm2D.backward called before forward")
        axes = (0, 2, 3)
        # grads for gamma/beta (optional keep if needed)
        self.dgamma = (grad_output * self.x_hat).sum(axis=axes, keepdims=True)
        self.dbeta  = grad_output.sum(axis=axes, keepdims=True)

        inv_std = cp.reciprocal(cp.sqrt(self.var + self.eps))
        dxhat = grad_output * self.gamma

        # BN training backward (per-channel)
        # dvar
        dvar = (dxhat * self.x_centered * (-0.5) * (inv_std**3)).sum(axis=axes, keepdims=True)
        # dmu
        dmu = (dxhat * (-inv_std)).sum(axis=axes, keepdims=True) + dvar * (-2.0 * self.x_centered).sum(axis=axes, keepdims=True) / self.M
        # dx
        dx = dxhat * inv_std + dvar * (2.0 * self.x_centered) / self.M + dmu / self.M
        return dx

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def stats(x, name):
    print(f"{name}: shape={tuple(x.shape)}, max={float(cp.max(cp.abs(x))):.3e}, "
          f"norm={float(cp.linalg.norm(x).astype(cp.float32)):.3e}")

# -------------------------------------------------------------------
# Mini model
#   Conv(16,3x3) -> ReLU -> BN -> Conv(8,3x3) -> ReLU -> Flatten -> Dense(10, ReLU)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Mini CNN: Conv → ReLU → BN → Conv → ReLU → Flatten → Dense ===")
    cp.random.seed(42)

    # Input
    x = cp.random.randn(8, 3, 32, 32).astype(cp.float32)  # NCHW

    # Layers
    conv1 = Conv2D(filters=16, kernel_size=(3,3), padding=(1,1), stride=(1,1), use_bias=True, initializer="he")
    relu1 = ReLU()
    bn1   = BatchNorm2D(num_channels=16, eps=1e-5)

    conv2 = Conv2D(filters=8,  kernel_size=(3,3), padding=(1,1), stride=(1,1), use_bias=True, initializer="he")
    relu2 = ReLU()

    flat  = Flatten()
    fc    = Dense(units=10, activation="relu", initializer="xavier", use_native_bwd=True)

    # (optional) build if your framework expects it
    conv1.build(x.shape)
    y1 = conv1(x)
    y1 = relu1(y1)
    y1 = bn1(y1)

    conv2.build(y1.shape)
    y2 = conv2(y1)
    y2 = relu2(y2)

    f = flat(y2)
    fc.build(f.shape)

    y3 = fc(f)

    print("\n--- Forward stats ---")
    stats(y1, "after BN1")
    stats(y2, "after Conv2+ReLU")
    stats(y3, "logits (Dense)")

    # Loss = sum(y3) → gY = ones
    gy3 = cp.ones_like(y3, dtype=cp.float32)

    # Backprop: Dense → Flatten → ReLU2 → Conv2 → BN → ReLU1 → Conv1
    g_f  = fc.backward(gy3)
    g_y2 = flat.backward(g_f)
    g_y1 = relu2.backward(g_y2)
    g_bn = conv2.backward(g_y1)
    g_bn = bn1.backward(g_bn)
    g_x  = relu1.backward(g_bn)
    g_x  = conv1.backward(g_x)

    print("\n--- Grad stats ---")
    stats(g_x,           "grad x")
    stats(conv1.dW,      "grad W_conv1")
    stats(conv2.dW,      "grad W_conv2")
    if conv1.db is not None: stats(conv1.db, "grad b_conv1")
    if conv2.db is not None: stats(conv2.db, "grad b_conv2")

    # Optional: BN parameter grads
    stats(bn1.dgamma, "grad gamma_bn1")
    stats(bn1.dbeta,  "grad beta_bn1")

    # Optional: Dense grads
    stats(fc.dW, "grad W_dense")
    if fc.db is not None: stats(fc.db, "grad b_dense")

    print("\n[OK] Mini CNN forward/backward completed ✅")
