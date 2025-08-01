# ✅ CuPy 기반 metrics 정의

import cupy as cp
# dev/backend/backend_ops/metrics/metrics.py

def mse(y_pred, y_true):
    return float(cp.mean((y_pred - y_true) ** 2))

def mae(y_pred, y_true):
    return float(cp.mean(cp.abs(y_pred - y_true)))

def accuracy(y_pred, y_true):
    pred_labels = cp.argmax(y_pred, axis=1)
    true_labels = cp.argmax(y_true, axis=1)
    return float(cp.mean(pred_labels == true_labels))
