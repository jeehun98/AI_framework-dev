
#include <cuda_runtime.h>
#include "backward_kernels.cuh"

__global__ void matmul_backward_input(const float* d_out, const float* W_T, float* d_input, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k)
            val += d_out[row * K + k] * W_T[k * N + col];
        d_input[row * N + col] = val;
    }
}

__global__ void matmul_backward_weight(const float* input_T, const float* d_out, float* d_weight, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < K) {
        float val = 0.0f;
        for (int m = 0; m < M; ++m)
            val += input_T[row * M + m] * d_out[m * K + col];
        d_weight[row * K + col] = val;
    }
}

__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i)
            sum += d_out[i * cols + col];
        d_bias[col] = sum;
    }
}
