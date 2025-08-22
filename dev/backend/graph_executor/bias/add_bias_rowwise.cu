// add_bias_rowwise.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do {                                       \
    cudaError_t _err = (expr);                                      \
    if (_err != cudaSuccess) {                                      \
        fprintf(stderr, "[CUDA] %s failed: %s (%s:%d)\n",           \
                #expr, cudaGetErrorString(_err), __FILE__, __LINE__);\
    }                                                               \
} while(0)
#endif

// ============================
// Row-wise bias: bias length == cols
// ============================
__global__ void add_bias_rowwise_kernel(const float* __restrict__ in,
                                        const float* __restrict__ bias, // [cols]
                                        float* __restrict__ out,
                                        int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = in[idx] + bias[col];
    }
}

// ============================
// Col-wise(채널) bias: bias length == rows_per_sample
// rows = batch * rows_per_sample
// ============================
__global__ void add_bias_colwise_kernel(const float* __restrict__ in,
                                        const float* __restrict__ bias, // [rows_per_sample]
                                        float* __restrict__ out,
                                        int rows, int cols, int rows_per_sample) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..rows-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..cols-1
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        int r_local = row % rows_per_sample;         // channel index within sample
        out[idx] = in[idx] + bias[r_local];
    }
}
void launch_add_bias_rowwise(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    add_bias_rowwise_kernel<<<grid, block, 0, stream>>>(in, bias, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_bias_colwise(const float* in, const float* bias, float* out,
                             int rows, int cols, int rows_per_sample,
                             cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    add_bias_colwise_kernel<<<grid, block, 0, stream>>>(
        in, bias, out, rows, cols, rows_per_sample);
    CUDA_CHECK(cudaGetLastError());
}