import numpy as np
import os
import sys

# ✅ test_setup.py 경로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/tests"))

from test_setup import import_cuda_module

# ✅ losses_cuda 모듈 import
losses_cuda = import_cuda_module(
    module_name="losses_cuda",
    build_dir="C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/losses/build/lib.win-amd64-cpython-312"
)

def test_mse_loss():
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    loss = losses_cuda.compute_loss(y_true, y_pred, "mse")
    expected = np.mean((y_true - y_pred) ** 2)

    print(f"[MSE Loss] CUDA: {loss:.6f}, NumPy: {expected:.6f}")
    assert abs(loss - expected) < 1e-6

def test_bce_loss():
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    expected = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    loss = losses_cuda.compute_loss(y_true, y_pred, "bce")

    print(f"[BCE Loss] CUDA: {loss:.6f}, NumPy: {expected:.6f}")
    assert abs(loss - expected) < 1e-6

if __name__ == "__main__":
    test_mse_loss()
    test_bce_loss()
    print("✅ 모든 손실 함수 테스트 통과")
