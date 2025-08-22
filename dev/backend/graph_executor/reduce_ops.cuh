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

// rows x cols 행 합산 → [cols] (열별 합)
__global__
void sum_rows_to_cols_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float acc = 0.f;
    // 한 쓰레드가 해당 열(col)의 모든 행을 순회
    for (int r = 0; r < rows; ++r) {
        acc += in[r * cols + col];
    }
    out[col] = acc;
}

inline void launch_reduce_over_rows(const float* in, float* out,
                                    int rows, int cols, cudaStream_t stream=0)
{
    // out은 완전히 덮어쓰므로 memset 필요 없음
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    sum_rows_to_cols_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// rows x cols 열 합산 → [rows] (행별 합)
__global__
void sum_cols_to_rows_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float acc = 0.f;
    for (int c = 0; c < cols; ++c) {
        acc += in[row * cols + c];
    }
    out[row] = acc;
}

inline void launch_reduce_over_cols(const float* in, float* out,
                                    int rows, int cols, cudaStream_t stream=0)
{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    sum_cols_to_rows_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// [rowsB] = [batch * rows_per_sample] 를 배치 방향으로 합쳐 [rows_per_sample]
__global__
void reduce_batch_stride_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int rows_per_sample,
                                int batch_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= rows_per_sample) return;
    float acc = 0.f;
    for (int b = 0; b < batch_size; ++b) {
        acc += in[b * rows_per_sample + k];
    }
    out[k] = acc;
}

inline void launch_reduce_batch_stride(const float* in, float* out,
                                       int rows_per_sample, int batch_size,
                                       cudaStream_t stream=0)
{
    dim3 block(256);
    dim3 grid((rows_per_sample + block.x - 1) / block.x);
    reduce_batch_stride_kernel<<<grid, block, 0, stream>>>(in, out, rows_per_sample, batch_size);
    CUDA_CHECK(cudaGetLastError());
}
