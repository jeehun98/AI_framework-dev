import os, sys
import numpy as np

# === Import path & CUDA DLL 경로 ===
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

np.random.seed(0)

def softmax_logits(X):
    x = X - X.max(axis=1, keepdims=True)
    p = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return p

def test_basic(M=4, N=7):
    X = np.random.randn(M, N).astype(np.float32)
    # 반드시 int32로!
    targets = np.random.randint(0, N, size=(M,), dtype=np.int32)

    # forward mean
    L = ge.cross_entropy(X, targets, reduction="mean")
    p = softmax_logits(X)
    ref = (-np.log(p[np.arange(M), targets])).mean()
    print("forward(mean) close:", np.allclose(L, ref, atol=1e-5))

    # backward mean
    dX = ge.cross_entropy_backward(X, targets, reduction="mean")
    ref_dX = p.copy()
    ref_dX[np.arange(M), targets] -= 1.0
    ref_dX /= M
    print("backward(mean) close:", np.allclose(dX, ref_dX, atol=1e-5))

def test_reductions(M=5, N=9):
    X = np.random.randn(M, N).astype(np.float32)
    targets = np.random.randint(0, N, size=(M,), dtype=np.int32)
    p = softmax_logits(X)

    # none
    L_vec = ge.cross_entropy(X, targets, reduction="none")
    ref_vec = -np.log(p[np.arange(M), targets])
    print("forward(none) close:", np.allclose(L_vec, ref_vec, atol=1e-5))

    # sum
    L_sum = ge.cross_entropy(X, targets, reduction="sum")
    ref_sum = ref_vec.sum()
    print("forward(sum) close:", np.allclose(L_sum, ref_sum, atol=1e-5))

def test_ignore_index(M=6, N=8):
    X = np.random.randn(M, N).astype(np.float32)
    targets = np.random.randint(0, N, size=(M,), dtype=np.int32)
    # 두 샘플을 무시하도록 index를 -1로 설정(=ignore_index)
    IGN = -1
    targets_ign = targets.copy()
    targets_ign[1] = IGN
    targets_ign[4] = IGN

    # forward mean with ignore_index
    L = ge.cross_entropy(X, targets_ign, reduction="mean", ignore_index=IGN)

    # numpy ref (유효 샘플만 평균)
    valid = (targets_ign != IGN)
    p = softmax_logits(X)
    ref = (-np.log(p[np.arange(M)[valid], targets_ign[valid]])).mean()
    print("forward(mean, ignore_index) close:", np.allclose(L, ref, atol=1e-5))

    # backward mean with ignore_index (무시된 행은 grad=0, 분모는 유효샘플수)
    dX = ge.cross_entropy_backward(X, targets_ign, reduction="mean", ignore_index=IGN)
    ref_dX = np.zeros_like(X, dtype=np.float32)
    Meff = valid.sum() if valid.sum() > 0 else 1
    ref_row = p.copy()
    ref_row[np.arange(M), targets] -= 1.0
    ref_row /= Meff
    ref_dX[valid] = ref_row[valid]
    # 무시된 행은 0
    print("backward(mean, ignore_index) close:", np.allclose(dX, ref_dX, atol=1e-5))

def test_label_smoothing(M=4, N=7, eps=0.1):
    X = np.random.randn(M, N).astype(np.float32)
    targets = np.random.randint(0, N, size=(M,), dtype=np.int32)
    p = softmax_logits(X)

    # forward (mean) with label smoothing
    L = ge.cross_entropy(X, targets, reduction="mean", label_smoothing=eps)
    # ref: (1-eps)*CE(onehot) + eps*CE(uniform)
    ce_onehot = -np.log(p[np.arange(M), targets]).mean()
    ce_uniform = -np.log(p).mean()  # uniform q -> 평균 log p
    ref = (1.0 - eps) * ce_onehot + eps * ce_uniform
    print("forward(mean, label_smoothing) close:", np.allclose(L, ref, atol=1e-5))

    # backward: dX = (p - q) / M, q = (1-eps)*onehot + eps/N
    q = np.full_like(p, eps / N, dtype=np.float32)
    q[np.arange(M), targets] += (1.0 - eps)
    dX = ge.cross_entropy_backward(X, targets, reduction="mean", label_smoothing=eps)
    ref_dX = (p - q) / M
    print("backward(mean, label_smoothing) close:", np.allclose(dX, ref_dX, atol=1e-5))

if __name__ == "__main__":
    test_basic()
    test_reductions()
    test_ignore_index()
    test_label_smoothing()
