#include "pooling_kernels.cuh"
#include <math_constants.h>  // CUDART_INF_F

__device__ __forceinline__ int clamp_i(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -------------------- Forward: MaxPool (NCHW) --------------------
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t* __restrict__ argmax,
    Pool2DParams p)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.N * p.C * p.H_out * p.W_out;
    if (idx >= total) return;

    int w_out = idx % p.W_out;
    int h_out = (idx / p.W_out) % p.H_out;
    int c     = (idx / (p.W_out * p.H_out)) % p.C;
    int n     =  idx / (p.W_out * p.H_out * p.C);

    int h_start = h_out * p.stride_h - p.pad_h;
    int w_start = w_out * p.stride_w - p.pad_w;

    float  best     = -CUDART_INF_F;
    int32_t best_ix = -1;

    const int in_c_stride = p.H * p.W;         // per-channel plane
    const int in_n_stride = p.C * in_c_stride; // per-sample

    for (int kh = 0; kh < p.kernel_h; ++kh) {
        int h = h_start + kh * p.dilation_h;
        if ((unsigned)h >= (unsigned)p.H) continue;

        for (int kw = 0; kw < p.kernel_w; ++kw) {
            int w = w_start + kw * p.dilation_w;
            if ((unsigned)w >= (unsigned)p.W) continue;

            int in_idx = n * in_n_stride + c * in_c_stride + h * p.W + w; // [n,c,h,w]
            float v = x[in_idx];
            if (v > best) { best = v; best_ix = in_idx; }
        }
    }
    y[idx] = best;
    if (argmax) argmax[idx] = best_ix;
}

// -------------------- Forward: AvgPool (NCHW) --------------------
__global__ void avgpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    Pool2DParams p)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.N * p.C * p.H_out * p.W_out;
    if (idx >= total) return;

    int w_out = idx % p.W_out;
    int h_out = (idx / p.W_out) % p.H_out;
    int c     = (idx / (p.W_out * p.H_out)) % p.C;
    int n     =  idx / (p.W_out * p.H_out * p.C);

    int h_start = h_out * p.stride_h - p.pad_h;
    int w_start = w_out * p.stride_w - p.pad_w;

    const int in_c_stride = p.H * p.W;
    const int in_n_stride = p.C * in_c_stride;

    float sum = 0.f;
    int   count = 0;
    const int win = p.kernel_h * p.kernel_w;

    for (int kh = 0; kh < p.kernel_h; ++kh) {
        int h = h_start + kh * p.dilation_h;
        for (int kw = 0; kw < p.kernel_w; ++kw) {
            int w = w_start + kw * p.dilation_w;

            bool inside = (h >= 0 && h < p.H && w >= 0 && w < p.W);
            if (inside) {
                int in_idx = n * in_n_stride + c * in_c_stride + h * p.W + w;
                sum += x[in_idx];
                ++count;
            } else if (p.avg_inclusive) {
                // include pad → add 0, still increment count
                ++count;
            }
        }
    }

    float denom = p.avg_inclusive ? (float)win : (float)count;
    y[idx] = (count == 0 ? 0.f : sum / denom);
}

// -------------------- Backward: MaxPool (NCHW) -------------------
__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    const int32_t* __restrict__ argmax,
    Pool2DParams p)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.N * p.C * p.H_out * p.W_out;
    if (idx >= total) return;

    int32_t in_idx = argmax ? argmax[idx] : -1;
    if (in_idx >= 0) {
        atomicAdd(grad_x + in_idx, grad_y[idx]);
    }
}

// -------------------- Backward: AvgPool (NCHW) -------------------
__global__ void avgpool2d_backward_kernel(
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    Pool2DParams p)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.N * p.C * p.H_out * p.W_out;
    if (idx >= total) return;

    int w_out = idx % p.W_out;
    int h_out = (idx / p.W_out) % p.H_out;
    int c     = (idx / (p.W_out * p.H_out)) % p.C;
    int n     =  idx / (p.W_out * p.H_out * p.C);

    int h_start = h_out * p.stride_h - p.pad_h;
    int w_start = w_out * p.stride_w - p.pad_w;

    const int in_c_stride = p.H * p.W;
    const int in_n_stride = p.C * in_c_stride;

    // recompute denominator the same way as forward
    int valid = 0;
    const int win = p.kernel_h * p.kernel_w;
    for (int kh = 0; kh < p.kernel_h; ++kh) {
        int h = h_start + kh * p.dilation_h;
        for (int kw = 0; kw < p.kernel_w; ++kw) {
            int w = w_start + kw * p.dilation_w;
            if (h >= 0 && h < p.H && w >= 0 && w < p.W) ++valid;
            else if (p.avg_inclusive) ++valid;
        }
    }
    float g = (valid == 0 ? 0.f : grad_y[idx] / (float)valid);

    for (int kh = 0; kh < p.kernel_h; ++kh) {
        int h = h_start + kh * p.dilation_h;
        for (int kw = 0; kw < p.kernel_w; ++kw) {
            int w = w_start + kw * p.dilation_w;
            if (h < 0 || h >= p.H || w < 0 || w >= p.W) continue;

            int in_idx = n * in_n_stride + c * in_c_stride + h * p.W + w;
            atomicAdd(grad_x + in_idx, g);
        }
    }
}
