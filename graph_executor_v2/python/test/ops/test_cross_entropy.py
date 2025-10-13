# test_cross_entropy.py
import os, sys, argparse
import numpy as np

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

from graph_executor_v2.ops import require

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None

ops_ce = require("cross_entropy")
Reduction = ops_ce.Reduction
CrossEntropyAttrs = ops_ce.CrossEntropyAttrs

def log_softmax(x, axis=-1):
    x = x.astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))

def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis=axis))

def one_hot(indices, num_classes):
    M = indices.shape[0]
    oh = np.zeros((M, num_classes), dtype=np.float32)
    oh[np.arange(M), indices] = 1.0
    return oh

def cross_entropy_ref(x, target, from_logits=True, reduction="mean",
                      ignore_index=-1, eps=1e-9, ls_eps=0.0):
    x = x.astype(np.float32)
    target = target.astype(np.int64)  # 내부 연산만 i64, 커널은 i32
    M, N = x.shape

    valid = (target != ignore_index)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return (np.zeros((M,), np.float32) if reduction=="none"
                else np.array([0.0], np.float32))

    tgt = np.clip(target, 0, N-1)
    q = one_hot(tgt, N)
    if ls_eps > 0.0:
        q = (1.0 - ls_eps) * q + (ls_eps / N)

    if from_logits:
        lsm = log_softmax(x, axis=1)
        loss_all = -np.sum(q * lsm, axis=1)
    else:
        p = np.clip(x, 1e-12, 1.0)
        loss_all = -np.sum(q * np.log(p + eps), axis=1)

    loss_all = np.where(valid, loss_all, 0.0)

    if reduction == "none":
        return loss_all.astype(np.float32)
    elif reduction == "sum":
        return np.array([np.sum(loss_all, dtype=np.float32)], dtype=np.float32)
    else:
        return np.array([np.sum(loss_all, dtype=np.float32) / n_valid], dtype=np.float32)

def cross_entropy_backward_ref(x, target, from_logits=True, reduction="mean",
                               ignore_index=-1, eps=1e-9, ls_eps=0.0):
    x = x.astype(np.float32)
    target = target.astype(np.int64)
    M, N = x.shape

    valid = (target != ignore_index)
    n_valid = max(int(np.sum(valid)), 1)

    tgt = np.clip(target, 0, N-1)
    q = one_hot(tgt, N)
    if ls_eps > 0.0:
        q = (1.0 - ls_eps) * q + (ls_eps / N)

    if from_logits:
        p = softmax(x, axis=1)
        dx = (p - q)
    else:
        p = np.clip(x, 1e-12, 1.0)
        dx = - q / (p + eps)

    dx = dx * valid[:, None].astype(np.float32)
    if reduction == "mean":
        dx /= n_valid
    return dx.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--from-logits", action="store_true")
    ap.add_argument("--reduction", choices=["none", "mean", "sum"], default="mean")
    ap.add_argument("--ignore-index", type=int, default=-1)
    ap.add_argument("--ls-eps", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args()

    print("LOADED:", ops_ce.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_cross_entropy expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)
    M, N = 5, 7

    if args.from_logits:
        x_h = rng.standard_normal(size=(M, N), dtype=np.float32)
    else:
        z = rng.standard_normal(size=(M, N), dtype=np.float32)
        p = np.exp(z - z.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        x_h = p.astype(np.float32)

    t_h = rng.integers(0, N, size=(M,), dtype=np.int32)  # ★ int32 고정
    if args.ignore_index >= 0:
        idxs = rng.choice(M, size=min(2, M), replace=False)
        t_h[idxs[0]] = args.ignore_index
        if len(idxs) > 1:
            t_h[idxs[1]] = args.ignore_index

    x_d  = cp.asarray(x_h)
    t_d  = cp.asarray(t_h)              # int32
    dx_d = cp.zeros_like(x_d)

    attrs = CrossEntropyAttrs()
    attrs.from_logits  = bool(args.from_logits)
    attrs.reduction    = getattr(Reduction, {"none":"None_","mean":"Mean","sum":"Sum"}[args.reduction])
    attrs.ignore_index = int(args.ignore_index)
    attrs.eps          = float(args.eps)
    attrs.ls_eps       = float(args.ls_eps)

    if args.reduction == "none":
        loss_d = cp.zeros((M,), dtype=cp.float32); loss_shape = [M]
    else:
        loss_d = cp.zeros((1,), dtype=cp.float32); loss_shape = [1]

    # Forward
    ops_ce.forward(
        int(x_d.data.ptr), [M, N],
        int(t_d.data.ptr), [M],
        int(loss_d.data.ptr), loss_shape,
        attrs, 0
    )
    loss_h = cp.asnumpy(loss_d)
    loss_ref = cross_entropy_ref(x_h, t_h, args.from_logits, args.reduction,
                                 args.ignore_index, args.eps, args.ls_eps)
    fwd_err = float(np.max(np.abs(loss_h - loss_ref)))
    print("Forward loss max_abs diff:", fwd_err)
    # assert fwd_err < 5e-5, f"forward mismatch: {fwd_err}"

    # Backward
    ops_ce.backward(
        int(x_d.data.ptr), [M, N],
        int(t_d.data.ptr), [M],
        int(dx_d.data.ptr), [M, N],
        attrs, 0
    )
    dx_h = cp.asnumpy(dx_d)
    dx_ref = cross_entropy_backward_ref(x_h, t_h, args.from_logits, args.reduction,
                                        args.ignore_index, args.eps, args.ls_eps)
    bwd_err = float(np.max(np.abs(dx_h - dx_ref)))
    print("Backward dX max_abs diff:", bwd_err)
    assert bwd_err < 8e-4, f"backward mismatch: {bwd_err}"

    print("OK: cross-entropy forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
