#pragma once

// ======= 손실 값 계산 (Forward) =======
float compute_mse_loss_cuda(float* y_true, float* y_pred, int size);
float compute_bce_loss_cuda(float* y_true, float* y_pred, int size);
float compute_cce_loss_cuda(float* y_true, float* y_pred, int batch_size, int num_classes);

// ======= 손실 함수의 기울기 계산 (Backward) =======
__global__ void mse_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int size);
__global__ void bce_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int size);
__global__ void cce_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int batch_size, int num_classes);
