// optimizer_kernels.cuh
#pragma once

#include "optimizer_types.cuh"

__global__ void sgd_kernel(float* param, float* grad, float lr, int size);
__global__ void momentum_kernel(float* param, float* grad, float* velocity, float lr, float beta, int size);
__global__ void adam_kernel(float* param, float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float epsilon, int t, int size);

void optimizer_update_cuda(
    float* param, float* grad,
    float* velocity, float* m, float* v,
    float lr, float beta1, float beta2, float epsilon,
    int size, OptimizerType opt_type, int t);
