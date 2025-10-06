# graph_executor_v2/python/test/layers/test_mini_cnn_train_standalone.py
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

# ---- minimal ReLU & BatchNorm2D (training mode) ----
class ReLU:
    def __init__(self): self.mask=None
    def __call__(self, x): return self.call(x)
    def call(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, gy):
        if self.mask is None: raise RuntimeError("ReLU.backward before forward")
        return gy * self.mask

class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5):
        self.C = int(num_channels); self.eps=float(eps)
        self.gamma = cp.ones((1,self.C,1,1), cp.float32)
        self.beta  = cp.zeros((1,self.C,1,1), cp.float32)
        self.x_hat=self.mu=self.var=self.x_centered=None
        self.M=None; self.dgamma=None; self.dbeta=None
    def __call__(self, x): return self.call(x)
    def call(self, x):
        axes=(0,2,3)
        self.mu  = x.mean(axis=axes, keepdims=True)
        self.var = x.var(axis=axes, keepdims=True)
        self.x_centered = x - self.mu
        inv_std = cp.reciprocal(cp.sqrt(self.var + self.eps))
        self.x_hat = self.x_centered * inv_std
        self.M = x.size // self.C
        return self.gamma * self.x_hat + self.beta
    def backward(self, gy):
        if any(v is None for v in (self.x_hat,self.mu,self.var,self.x_centered,self.M)):
            raise RuntimeError("BN.backward before forward")
        axes=(0,2,3)
        self.dgamma = (gy * self.x_hat).sum(axis=axes, keepdims=True)
        self.dbeta  = gy.sum(axis=axes, keepdims=True)
        inv_std = cp.reciprocal(cp.sqrt(self.var + self.eps))
        dxhat = gy * self.gamma
        dvar = (dxhat * self.x_centered * (-0.5) * (inv_std**3)).sum(axis=axes, keepdims=True)
        dmu  = (dxhat * (-inv_std)).sum(axis=axes, keepdims=True) + dvar * (-2.0*self.x_centered).sum(axis=axes, keepdims=True)/self.M
        dx   = dxhat * inv_std + dvar * (2.0*self.x_centered)/self.M + dmu/self.M
        return dx

# ---- utils ----
def softmax_logits_to_prob(logits):
    # logits: (N, C)
    z = logits - logits.max(axis=1, keepdims=True)
    expz = cp.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def cross_entropy_prob(p, y_idx):
    # p: (N,C), y_idx: (N,) int
    N = p.shape[0]
    logp = cp.log(p[cp.arange(N), y_idx] + 1e-12)
    return -logp.mean()

def ce_grad_logits(logits, y_idx):
    # dL/dlogits = (p - onehot) / N
    N, C = logits.shape
    p = softmax_logits_to_prob(logits)
    gy = p
    gy[cp.arange(N), y_idx] -= 1.0
    gy /= N
    return gy

def stats(x, name):
    print(f"{name}: shape={tuple(x.shape)}, max={float(cp.max(cp.abs(x))):.3e}, "
          f"norm={float(cp.linalg.norm(x).astype(cp.float32)):.3e}")

# ---- training demo ----
if __name__ == "__main__":
    print("=== Mini CNN Train: Conv→ReLU→BN→Conv→ReLU→Flatten→Dense (Softmax CE) ===")
    cp.random.seed(7)

    # data (one minibatch)
    N, C, H, W = 16, 3, 32, 32
    num_classes = 10
    x = cp.random.randn(N, C, H, W).astype(cp.float32)
    y = cp.random.randint(0, num_classes, size=(N,), dtype=cp.int32)

    # model
    conv1 = Conv2D(filters=16, kernel_size=(3,3), padding=(1,1), use_bias=True, initializer="he")
    relu1 = ReLU()
    bn1   = BatchNorm2D(num_channels=16)

    conv2 = Conv2D(filters=8, kernel_size=(3,3), padding=(1,1), use_bias=True, initializer="he")
    relu2 = ReLU()

    flat  = Flatten()
    # Dense는 logits이 필요하므로 activation='none'
    fc    = Dense(units=num_classes, activation="none", initializer="xavier", use_native_bwd=True)

    # optional build
    conv1.build(x.shape)
    h1 = conv1(x); h1 = relu1(h1); h1 = bn1(h1)
    conv2.build(h1.shape)
    h2 = conv2(h1); h2 = relu2(h2)
    f  = flat(h2)
    fc.build(f.shape)
    logits = fc(f)

    # training hyperparams
    lr = 1e-2
    steps = 20

    def step_sgd(lr):
        # forward
        h1 = conv1(x); h1 = relu1(h1); h1 = bn1(h1)
        h2 = conv2(h1); h2 = relu2(h2)
        f  = flat(h2)
        logits = fc(f)  # (N, num_classes)

        # loss
        p = softmax_logits_to_prob(logits)
        loss = cross_entropy_prob(p, y)

        # backward (dL/dlogits)
        gy = ce_grad_logits(logits, y)
        g_f  = fc.backward(gy)
        g_h2 = flat.backward(g_f)
        g_h1 = relu2.backward(g_h2)
        g_bn = conv2.backward(g_h1)
        g_bn = bn1.backward(g_bn)
        g_h0 = relu1.backward(g_bn)
        _    = conv1.backward(g_h0)

        # SGD updates
        # Conv1
        conv1.W[...] -= lr * conv1.dW
        if conv1.db is not None: 
            conv1.db[...] -= lr * conv1.db

        # BN1
        bn1.gamma[...] -= lr * bn1.dgamma
        bn1.beta[...]  -= lr * bn1.dbeta
        # Conv2
        conv2.W[...] -= lr * conv2.dW
        if conv2.db is not None: 
            conv2.db[...] -= lr * conv2.db

        # Dense
        fc.W[...] -= lr * fc.dW
        if fc.db is not None: 
            fc.b[...] -= lr * fc.db

        return float(loss), logits

    # train loop
    loss_hist = []
    for t in range(steps):
        loss, logits = step_sgd(lr)
        loss_hist.append(loss)
        if (t % 2) == 0 or t == steps-1:
            pred = logits.argmax(axis=1)
            acc = float((pred == y).mean())
            print(f"step {t:02d} | loss={loss:.4f} | acc={acc:.3f}")

    print("\n--- Loss trend ---")
    print(" → ".join(f"{l:.4f}" for l in loss_hist[:5]), "...", f"{loss_hist[-1]:.4f}")

    # final stats
    stats(conv1.dW, "dW_conv1"); stats(conv2.dW, "dW_conv2"); stats(fc.dW, "dW_dense")
    print("\n[OK] Mini CNN training demo finished ✅")
