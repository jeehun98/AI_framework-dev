# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))                      # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))           # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------


import cupy as cp
from graph_executor_v2.layers.softmax_ce import SoftmaxCrossEntropy

def main():
    cp.random.seed(3)
    M, N = 10, 7
    logits  = cp.random.randn(M, N).astype(cp.float32)
    targets = cp.random.randint(0, N, size=(M,), dtype=cp.int32)

    # per-sample loss (reduction='none')
    ce = SoftmaxCrossEntropy(label_smoothing=0.05, reduction="none", from_logits=True)
    loss_vec = ce((logits, targets))
    assert loss_vec.shape == (M,), f"loss_vec.shape={loss_vec.shape}"
    print(f"[SoftmaxCE] forward OK: loss per-sample shape {loss_vec.shape}")

    # mean loss를 가정한 dL/dlogits = ce.backward(1/M)
    grad_scale = cp.full((M,), 1.0 / M, dtype=cp.float32)
    dlogits = ce.backward(grad_scale)
    assert dlogits.shape == logits.shape, f"dlogits.shape={dlogits.shape}"
    print("[SoftmaxCE] backward OK")

    # (옵션) reduction='mean' 경로도 점검
    ce_mean = SoftmaxCrossEntropy(label_smoothing=0.0, reduction="mean", from_logits=True)
    loss_mean = ce_mean((logits, targets))
    assert loss_mean.shape == (1,), f"loss_mean.shape={loss_mean.shape}"
    dlogits_mean = ce_mean.backward(cp.array(1.0, dtype=cp.float32))  # scalar scale
    assert dlogits_mean.shape == logits.shape
    print("[SoftmaxCE] reduction='mean' OK")

if __name__ == "__main__":
    main()
    print("[SoftmaxCE] all good ✅")
