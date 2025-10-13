# python/graph_executor_v2/losses/utils.py
from __future__ import annotations
import cupy as cp

def infer_grad_scale(loss_fn, model, X, y):
    """
    dY 스케일(sum/mean)을 감지해 권장 grad_scale을 반환.
    - sum 스케일: ||dY||_1 ~ O(B) → 1/B 권장
    - mean 스케일: ||dY||_1 ~ O(1) → 1.0 권장
    """
    logits = model(X)
    _, dY = loss_fn.forward(logits, y)
    B = X.shape[0]
    s_sum = float(cp.abs(dY).sum())
    if s_sum > 10.0 and (s_sum / max(1e-6, B)) > 0.05:
        return 1.0 / B, "sum"
    return 1.0, "mean"
