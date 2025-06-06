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

# ✅ 손실 함수 테스트
def test_mse_loss():
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    loss = losses_cuda.mse_loss(y_true, y_pred)
    expected = np.mean((y_true - y_pred) ** 2)

    print(f"[MSE Loss] CUDA: {loss:.6f}, NumPy: {expected:.6f}")
    assert abs(loss - expected) < 1e-6

# ✅ MSE Gradient 테스트
def test_mse_grad():
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    grad_cuda = losses_cuda.mse_grad(y_true, y_pred)
    grad_numpy = 2 * (y_pred - y_true) / y_true.size

    print(f"[MSE Grad] CUDA: {grad_cuda}, NumPy: {grad_numpy}")
    assert np.allclose(grad_cuda, grad_numpy, atol=1e-6)

# ✅ BCE 손실 테스트 (grad는 아직 미지원)
def test_bce_loss():
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    expected = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    loss = losses_cuda.binary_crossentropy(y_true, y_pred)

    print(f"[BCE Loss] CUDA: {loss:.6f}, NumPy: {expected:.6f}")
    assert abs(loss - expected) < 1e-6

if __name__ == "__main__":
    test_mse_loss()
    test_mse_grad()
    test_bce_loss()
    print("✅ 모든 손실 및 gradient 테스트 통과")
