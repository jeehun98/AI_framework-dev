// pooling_ops.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "pooling_kernels.cuh"  // Pool2DParams, __global__ kernels

// Host wrappers (stream 인자 포함)
void maxpool2d_forward(
    const float* x, float* y, int32_t* argmax,
    const Pool2DParams& p, cudaStream_t stream = 0);

void avgpool2d_forward(
    const float* x, float* y,
    const Pool2DParams& p, cudaStream_t stream = 0);

void maxpool2d_backward(
    const float* grad_y, float* grad_x, const int32_t* argmax,
    const Pool2DParams& p, cudaStream_t stream = 0);

void avgpool2d_backward(
    const float* grad_y, float* grad_x,
    const Pool2DParams& p, cudaStream_t stream = 0);
