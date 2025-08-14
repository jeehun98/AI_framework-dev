#include "add_bias_rowwise.cuh"
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

// 2D 인덱싱으로 모듈러 연산 제거, coalesced 접근 유지
static __global__ void add_bias_rowwise_kernel(const float* __restrict__ in,
                                               const float* __restrict__ bias, // [cols]
                                               float* __restrict__ out,
                                               int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        out[idx] = in[idx] + bias[col];
    }
}

void launch_add_bias_rowwise(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream) {
    // warp가 x축으로 길게 뻗도록 구성 → 연속 주소 접근
    dim3 block(32, 8); // 256 threads
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    add_bias_rowwise_kernel<<<grid, block, 0, stream>>>(in, bias, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
