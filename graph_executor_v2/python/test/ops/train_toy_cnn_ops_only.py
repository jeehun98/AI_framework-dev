# train_toy_cnn_ops_only.py
import os, sys, math, time
import numpy as np

# ============== Configs ==============
class CFG:
    # data
    H = 16
    W = 16
    in_channels = 1
    num_classes = 2

    # conv
    conv_out = 8
    k = 3
    stride = 1
    pad = 1
    dil = 1
    groups = 1

    # pool
    pool_k = 2
    pool_s = 2
    pool_p = 0
    pool_d = 1
    ceil_mode = False

    # training
    epochs = 8
    batch = 64
    iters_per_epoch = 50
    base_lr = 0.2
    weight_decay = 1e-4          # decoupled weight decay (AdamW style)
    betas = (0.9, 0.999)
    eps = 1e-8
    grad_clip = 5.0              # set 0.0 to disable

    # reproducibility
    seed = 0

# ============== Import path & CUDA DLL 경로 (Windows) ==============
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

# ============== CuPy ==============
try:
    import cupy as cp
except Exception as e:
    print("CuPy is required for this demo:", e)
    sys.exit(1)

# ============== Load our independent ops ==============
from graph_executor_v2.ops import require
ops_conv2d = require("conv2d")     # -> _ops_conv2d
Conv2DAttrs = ops_conv2d.Conv2DAttrs

ops_pool2d = require("pool2d")     # -> _ops_pool2d
Pool2DAttrs = ops_pool2d.Pool2DAttrs

ops_gemm = require("gemm")         # -> _ops_gemm (저수준 GEMM 사용)

# ------------------------------------------------------------
# Utils: dataset
# ------------------------------------------------------------
def make_toy_batch(batch, H=16, W=16, seed=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((batch, 1, H, W), dtype=np.float32)
    y = np.zeros((batch,), dtype=np.int64)

    cy, cx = H // 2, W // 2
    rr, cc = np.ogrid[:H, :W]
    center_mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= 2
    border_mask = (rr == 0) | (rr == H-1) | (cc == 0) | (cc == W-1)

    for i in range(batch):
        if rng.random() < 0.5:
            X[i, 0, center_mask] = 1.0
            y[i] = 0
        else:
            X[i, 0, border_mask] = 1.0
            y[i] = 1
        X[i, 0] += rng.normal(0, 0.05, size=(H, W)).astype(np.float32)
        X[i, 0] = np.clip(X[i, 0], 0.0, 1.0)
    return X, y

# ------------------------------------------------------------
# Utils: math ops
# ------------------------------------------------------------
def softmax_logits(logits: "cp.ndarray"):
    # logits: (N, C)
    m = logits.max(axis=1, keepdims=True)
    e = cp.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def nll_loss(probs, targets):
    # targets: int64 (N,), probs: (N, C)
    N = probs.shape[0]
    idx = (cp.arange(N), targets)
    return -cp.log(probs[idx] + 1e-12).mean()

def accuracy(probs, targets):
    pred = probs.argmax(axis=1)
    return float((pred == targets).mean())

def relu_forward(x):
    return cp.maximum(x, 0)

def relu_backward(gy, x):
    mask = (x > 0).astype(gy.dtype)
    return gy * mask

def cosine_lr(base_lr, t, T):
    # t: 0..T-1 (epoch index), T: total epochs
    if T <= 1:
        return base_lr
    return 0.5 * base_lr * (1 + math.cos(math.pi * t / (T - 1)))

def clip_grad_norm_(grads, max_norm):
    if max_norm is None or max_norm <= 0:
        return 0.0
    # grads: list of cp.ndarray
    total = cp.array(0.0, dtype=cp.float32)
    for g in grads:
        if g is not None:
            total += (g * g).sum()
    total = cp.sqrt(total)
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            if g is not None:
                g *= scale
    return float(total.get())

# ------------------------------------------------------------
# GEMM helpers (독립 모듈만 사용)
# ------------------------------------------------------------
def _ptr(x: "cp.ndarray") -> int:
    return int(x.data.ptr)

def _t2d(x: "cp.ndarray", shape=None):
    """CuPy 배열을 ai::Tensor(2D)로 래핑."""
    if shape is None:
        assert x.ndim == 2, "shape 생략시 2D여야 합니다"
        shape = list(x.shape)
    return ops_gemm.make_tensor_2d(_ptr(x), list(map(int, shape)))

def linear_forward(A: "cp.ndarray", W: "cp.ndarray", B: "cp.ndarray"):
    """
    독립 모듈 GEMM(device)만 사용.
    A: (N,K), W: (K,C), B: (C,)
    return: logits Y (N,C)
    """
    N, K = A.shape
    K2, C = W.shape
    assert K == K2

    # 출력 버퍼
    Y = cp.empty((N, C), dtype=cp.float32)
    Bias2d = B.reshape(1, C)

    # attrs
    attrs = ops_gemm.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.act = getattr(ops_gemm.ActKind, "None")
    attrs.with_bias = True
    attrs.leaky_slope = 0.0

    # ai::Tensor 래핑
    A_t   = _t2d(A, [N, K])
    W_t   = _t2d(W, [K, C])
    Bias_t= _t2d(Bias2d, [1, C])
    Y_t   = _t2d(Y, [N, C])

    # 호출
    ops_gemm.forward(A_t, W_t, Bias_t, Y_t, attrs, None)
    return Y

# ------------------------------------------------------------
# Optimizer: AdamW (decoupled)
# ------------------------------------------------------------
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0):
        self.params = params  # list of dict: {"p": cp.ndarray, "g": cp.ndarray or None, "name": str}
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        # states
        self.m = [cp.zeros_like(p["p"]) for p in params]
        self.v = [cp.zeros_like(p["p"]) for p in params]

    def step(self, lr_override=None):
        self.t += 1
        lr = self.lr if lr_override is None else lr_override
        b1, b2, eps, wd = self.beta1, self.beta2, self.eps, self.wd

        for i, slot in enumerate(self.params):
            p = slot["p"]
            g = slot.get("g", None)
            if g is None:
                continue

            # decoupled weight decay
            if wd > 0.0:
                p -= lr * wd * p

            m = self.m[i]
            v = self.v[i]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)

            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p -= lr * m_hat / (cp.sqrt(v_hat) + eps)

    def zero_grad(self):
        for slot in self.params:
            if "g" in slot and slot["g"] is not None:
                slot["g"].fill(0)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class ToyCNN:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)

        # Conv params
        self.Cin = CFG.in_channels
        self.Cout = CFG.conv_out
        self.Kh = self.Kw = CFG.k

        w_std = math.sqrt(2.0 / (self.Cin * self.Kh * self.Kw))
        self.Wc = cp.asarray(rng.normal(0, w_std, size=(self.Cout, self.Cin, self.Kh, self.Kw)).astype(np.float32))
        self.Bc = cp.zeros((self.Cout,), dtype=cp.float32)

        # Spatial
        self.H = CFG.H
        self.W = CFG.W
        # after pooling 2x2 stride2 → H/2, W/2
        self.Ho = self.H // 2
        self.Wo = self.W // 2

        # Linear params
        self.flat_dim = self.Cout * self.Ho * self.Wo
        self.num_classes = CFG.num_classes
        wl_std = math.sqrt(2.0 / self.flat_dim)
        self.Wl = cp.asarray(rng.normal(0, wl_std, size=(self.flat_dim, self.num_classes)).astype(np.float32))
        self.Bl = cp.zeros((self.num_classes,), dtype=cp.float32)

        # attrs
        self.conv_attrs = Conv2DAttrs()
        self.conv_attrs.stride_h = self.conv_attrs.stride_w = CFG.stride
        self.conv_attrs.pad_h = self.conv_attrs.pad_w = CFG.pad
        self.conv_attrs.dil_h = self.conv_attrs.dil_w = CFG.dil
        self.conv_attrs.groups = CFG.groups

        self.pool_attrs = Pool2DAttrs()
        self.pool_attrs.kH = self.pool_attrs.kW = CFG.pool_k
        self.pool_attrs.sH = self.pool_attrs.sW = CFG.pool_s
        self.pool_attrs.pH = self.pool_attrs.pW = CFG.pool_p
        self.pool_attrs.dH = self.pool_attrs.dW = CFG.pool_d
        self.pool_attrs.ceil_mode = CFG.ceil_mode

        # optimizer (AdamW)
        self.opt = AdamW(
            params=[
                {"name": "Wc", "p": self.Wc, "g": None},
                {"name": "Bc", "p": self.Bc, "g": None},
                {"name": "Wl", "p": self.Wl, "g": None},
                {"name": "Bl", "p": self.Bl, "g": None},
            ],
            lr=CFG.base_lr,
            betas=CFG.betas,
            eps=CFG.eps,
            weight_decay=CFG.weight_decay,
        )

    # -------- forward --------
    def forward(self, Xd):
        N, Cin, H, W = Xd.shape
        assert Cin == self.Cin and H == self.H and W == self.W

        # Conv FWD
        Yc = cp.zeros((N, self.Cout, H, W), dtype=cp.float32)
        ops_conv2d.forward(
            int(Xd.data.ptr), [N, Cin, H, W],
            int(self.Wc.data.ptr), [self.Cout, self.Cin, self.Kh, self.Kw],
            int(Yc.data.ptr), [N, self.Cout, H, W],
            int(self.Bc.data.ptr),
            self.conv_attrs, 0
        )
        Zc = Yc
        Rc = relu_forward(Zc)

        # Pool FWD (with indices)
        Ho, Wo = self.Ho, self.Wo
        Yp = cp.empty((N, self.Cout, Ho, Wo), dtype=cp.float32)
        Ind = cp.empty((N, self.Cout, Ho, Wo), dtype=cp.int32)
        ops_pool2d.maxpool2d_forward(
            int(Rc.data.ptr), [N, self.Cout, H, W],
            int(Yp.data.ptr), [N, self.Cout, Ho, Wo],
            int(Ind.data.ptr),
            self.pool_attrs, 0
        )

        # Flatten → Linear (GEMM 독립 모듈)
        A = Yp.reshape(N, self.flat_dim)  # (N, flat_dim)
        logits = linear_forward(A, self.Wl, self.Bl)

        ctx = dict(
            X=Xd, Zc=Zc, Rc=Rc, Ind=Ind, A=A, logits=logits
        )
        return logits, ctx

    # -------- backward & update --------
    def backward(self, ctx, targets, lr_now=None):
        logits = ctx["logits"]
        N = logits.shape[0]
        probs = softmax_logits(logits)
        loss = nll_loss(probs, targets)
        acc = accuracy(probs, targets)

        # dL/dlogits
        gy = probs
        gy[cp.arange(N), targets] -= 1.0
        gy /= N  # mean

        # -------- Linear grads via _ops_gemm.backward --------
        A = ctx["A"]                 # (N, flat_dim)
        N, C = gy.shape              # C == num_classes
        flat_dim = A.shape[1]

        gA  = cp.empty_like(A)                 # (N, flat_dim)
        gWl = cp.empty_like(self.Wl)           # (flat_dim, C)
        gBl = cp.empty_like(self.Bl)           # (C,)
        Bl2d  = self.Bl.reshape(1, C)
        gBl2d = gBl.reshape(1, C)

        attrs = ops_gemm.GemmAttrs()
        attrs.trans_a = False
        attrs.trans_b = False
        attrs.act = getattr(ops_gemm.ActKind, "None")
        attrs.with_bias = True
        attrs.leaky_slope = 0.0

        A_t    = _t2d(A,            [N, flat_dim])
        Wl_t   = _t2d(self.Wl,      [flat_dim, C])
        Y_t    = _t2d(ctx["logits"],[N, C])      # act=None → Z==Y
        gY_t   = _t2d(gy,           [N, C])
        gA_t   = _t2d(gA,           [N, flat_dim])
        gWl_t  = _t2d(gWl,          [flat_dim, C])
        Bl_t   = _t2d(Bl2d,         [1, C])
        gBl_t  = _t2d(gBl2d,        [1, C])

        # (A,B,?, gY, Z, gA, gB, ?, gBias, attrs, stream)
        ops_gemm.backward(A_t, Wl_t, None, gY_t, Y_t,
                          gA_t, gWl_t, None, gBl_t,
                          attrs, None)

        # reshape back to pooled shape
        N, C, Ho, Wo = ctx["Rc"].shape[0], self.Cout, self.Ho, self.Wo
        gYp = gA.reshape(N, C, Ho, Wo)

        # Pool BWD
        dRc = cp.zeros_like(ctx["Rc"])
        ops_pool2d.maxpool2d_backward(
            int(gYp.data.ptr), [N, C, Ho, Wo],
            int(ctx["Ind"].data.ptr), [N, C, Ho, Wo],
            int(dRc.data.ptr), [N, C, self.H, self.W],
            self.pool_attrs, 0
        )

        # ReLU BWD
        dZc = relu_backward(dRc, ctx["Zc"])

        # Conv BWD
        dWc = cp.zeros_like(self.Wc)
        dBc = cp.zeros_like(self.Bc)
        dX  = cp.zeros_like(ctx["X"])
        ops_conv2d.backward(
            int(ctx["X"].data.ptr), [N, self.Cin, self.H, self.W],
            int(self.Wc.data.ptr), [self.Cout, self.Cin, self.Kh, self.Kw],
            int(dZc.data.ptr),     [N, self.Cout, self.H, self.W],
            int(dWc.data.ptr),     # dW
            int(dBc.data.ptr),     # dB
            int(dX.data.ptr),      # dX
            self.conv_attrs, 0
        )

        # --------- optimizer step (AdamW + grad clip) ----------
        self.opt.params[0]["g"] = dWc
        self.opt.params[1]["g"] = dBc
        self.opt.params[2]["g"] = gWl
        self.opt.params[3]["g"] = gBl

        if CFG.grad_clip and CFG.grad_clip > 0:
            _ = clip_grad_norm_([dWc, dBc, gWl, gBl], CFG.grad_clip)

        self.opt.step(lr_override=lr_now)
        self.opt.zero_grad()

        return float(loss), float(acc)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def main():
    # Reproducibility
    if CFG.seed is not None:
        np.random.seed(CFG.seed)

    np.set_printoptions(precision=4, suppress=True)
    print("LOADED ops:",
          getattr(ops_conv2d, "__file__", str(ops_conv2d)),
          getattr(ops_pool2d, "__file__", str(ops_pool2d)),
          getattr(ops_gemm, "__file__", str(ops_gemm)),
          sep="\n  ")

    model = ToyCNN(seed=CFG.seed)
    rng = np.random.default_rng(CFG.seed)

    # quick hold-out eval set
    X_eval_h, y_eval_h = make_toy_batch(512, H=CFG.H, W=CFG.W, seed=1234)
    X_eval = cp.asarray(X_eval_h)
    y_eval = cp.asarray(y_eval_h)

    for ep in range(1, CFG.epochs + 1):
        # LR schedule
        lr_now = cosine_lr(CFG.base_lr, ep - 1, CFG.epochs)

        losses, accs = [], []
        t0 = time.time()
        for _ in range(CFG.iters_per_epoch):
            X_h, y_h = make_toy_batch(CFG.batch, H=CFG.H, W=CFG.W,
                                      seed=rng.integers(1 << 31))
            Xd = cp.asarray(X_h)          # (N,1,16,16)
            yd = cp.asarray(y_h)          # (N,)

            logits, ctx = model.forward(Xd)
            loss, acc = model.backward(ctx, yd, lr_now=lr_now)
            losses.append(loss); accs.append(acc)

        # epoch summary
        dt = time.time() - t0
        train_loss = float(np.mean(losses))
        train_acc = float(np.mean(accs))

        # eval
        logits_eval, _ = model.forward(X_eval)
        probs_eval = softmax_logits(logits_eval)
        eval_acc = accuracy(probs_eval, y_eval)

        print(f"[epoch {ep:02d}] "
              f"lr={lr_now:.4f}  "
              f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
              f"eval_acc={eval_acc:.3f}  "
              f"({dt:.2f}s)")

    # final eval on fresh set
    X_h, y_h = make_toy_batch(256, H=CFG.H, W=CFG.W, seed=4321)
    Xd = cp.asarray(X_h); yd = cp.asarray(y_h)
    logits, _ = model.forward(Xd)
    probs = softmax_logits(logits)
    acc = accuracy(probs, yd)
    print(f"Final eval acc: {acc:.3f}")

if __name__ == "__main__":
    main()
