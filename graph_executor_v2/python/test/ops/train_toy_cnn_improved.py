# train_toy_cnn_improved.py
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

    # schedules / modes
    cosine_lr = True             # cosine anneal per epoch
    use_ops_gemm_numpy_fallback = True  # fallback to CPU ops_gemm.forward_numpy
    try_ops_gemm_device_first = False   # 만약 _ops_gemm 의 GPU forward 바인딩이 준비됐다면 True로

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

try:
    ops_gemm = require("gemm")     # -> _ops_gemm (numpy/device 여부는 런타임 체크)
except Exception:
    ops_gemm = None

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
# Linear (GEMM) forward helpers
# ------------------------------------------------------------
def linear_forward(A: "cp.ndarray", W: "cp.ndarray", B: "cp.ndarray"):
    """
    우선 CuPy GEMM (GPU)로 실행.
    옵션에 따라 _ops_gemm.forward_numpy 로 폴백 가능.
    필요 시 _ops_gemm 의 GPU forward (바인딩 준비됐을 경우)로 먼저 시도.
    """
    N, K = A.shape
    K2, C = W.shape
    assert K == K2, f"linear shape mismatch: A({N},{K}) x W({K2},{C})"

    # 1) (선택) _ops_gemm 의 GPU forward 가 있다면 먼저 시도
    if CFG.try_ops_gemm_device_first and ops_gemm is not None and hasattr(ops_gemm, "forward"):
        try:
            Y = cp.empty((N, C), dtype=cp.float32)
            # 아래 시그니처는 conv/pool 과 유사하다고 가정 (실제 바인딩에 맞춰 조정 필요)
            ops_gemm.forward(
                int(A.data.ptr), [N, K],
                int(W.data.ptr), [K, C],
                int(Y.data.ptr), [N, C],
                int(B.data.ptr),
                0  # stream
            )
            return Y
        except Exception:
            # 실패 시 CuPy로 후퇴
            pass

    # 2) CuPy GEMM (권장 경로: 완전 GPU)
    Y = A @ W  # (N, C)
    if B is not None:
        Y += B[cp.newaxis, :]
    return Y

def linear_forward_numpy_fallback(A: "cp.ndarray", W: "cp.ndarray", B: "cp.ndarray"):
    """CPU 경로: ops_gemm.forward_numpy 사용 (디버그/비교용)"""
    if ops_gemm is None or not hasattr(ops_gemm, "forward_numpy"):
        raise RuntimeError("ops_gemm.forward_numpy is not available")
    A_h = cp.asnumpy(A)
    W_h = cp.asnumpy(W)
    B_h = cp.asnumpy(B) if B is not None else None
    logits_h = ops_gemm.forward_numpy(A_h, W_h, B_h, act="none")
    return cp.asarray(logits_h)

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

        # Flatten → Linear
        A = Yp.reshape(N, self.flat_dim)  # (N, 512)
        logits = linear_forward(A, self.Wl, self.Bl)

        # (선택) CPU 경로 비교/디버그
        if CFG.use_ops_gemm_numpy_fallback:
            try:
                logits_np = linear_forward_numpy_fallback(A, self.Wl, self.Bl)
                # 혼동 방지: 실사용은 GPU logits, CPU는 비교만
                # diff = float(cp.max(cp.abs(logits - logits_np)).get())
                # if diff > 1e-4:
                #     print(f"[warn] CuPy vs ops_gemm.forward_numpy max diff={diff:.2e}")
                logits = logits  # keep GPU path
            except Exception:
                pass

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

        # Linear grads
        A = ctx["A"]  # (N, 512)
        gWl = A.T @ gy                  # (512, 2)
        gBl = gy.sum(axis=0)            # (2,)
        gA  = gy @ self.Wl.T            # (N, 512)

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
        # pack grads to optimizer slots
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
          getattr(ops_gemm, "__file__", str(ops_gemm)) if ops_gemm else "(gemm module not found)",
          sep="\n  ")

    model = ToyCNN(seed=CFG.seed)
    rng = np.random.default_rng(CFG.seed)

    # quick hold-out eval set
    X_eval_h, y_eval_h = make_toy_batch(512, H=CFG.H, W=CFG.W, seed=1234)
    X_eval = cp.asarray(X_eval_h)
    y_eval = cp.asarray(y_eval_h)

    for ep in range(1, CFG.epochs + 1):
        # LR schedule
        lr_now = CFG.base_lr
        if CFG.cosine_lr:
            lr_now = cosine_lr(CFG.base_lr, ep - 1, CFG.epochs)

        losses, accs = [], []
        t0 = time.time()
        for it in range(CFG.iters_per_epoch):
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
        with cp.cuda.Device():
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
