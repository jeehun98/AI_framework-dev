#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// NCHW layout parameters
struct Pool2DParams {
    int N, C, H, W;        // input shape
    int H_out, W_out;      // output spatial
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    bool avg_inclusive;    // AvgPool: include padding in denominator
};

// (optional) small device helper (can be used by kernels)
__device__ __forceinline__ int clamp_i(int x, int lo, int hi);

// ---- CUDA kernels (NCHW) ----
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t* __restrict__ argmax,
    Pool2DParams p);

__global__ void avgpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    Pool2DParams p);

__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    const int32_t* __restrict__ argmax,
    Pool2DParams p);

__global__ void avgpool2d_backward_kernel(
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    Pool2DParams p);
