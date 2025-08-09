#pragma once
#include <cuda_runtime.h>

// ======= 손실 값 계산 (Forward) =======
float compute_mse_loss_cuda(float* y_true, float* y_pred, int size);
float compute_bce_loss_cuda(float* y_true, float* y_pred, int size);
float compute_cce_loss_cuda(float* y_true, float* y_pred, int batch_size, int num_classes);

// ======= 손실 함수의 기울기 계산 (Backward) =======
__global__ void mse_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int size);


__global__ void bce_loss_backward(
    const float* __restrict__ y_true,   // [size]
    const float* __restrict__ y_pred,   // [size] (Sigmoid 출력 a)
    float* __restrict__ grad_out,       // [size] (dL/da)
    int size,                           // total elements = batch_size * elems_per_sample
    int batch_size                      // 평균 분모(배치 평균)
);

    __global__ void cce_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int batch_size, int num_classes);
