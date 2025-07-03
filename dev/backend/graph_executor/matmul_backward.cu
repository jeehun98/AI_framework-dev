#include "matmul_backward.cuh"

// d_input = d_out × Wᵗ
__global__ void matmul_backward_input_kernel(const float* d_out, const float* W, float* d_input,
                                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // K

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += d_out[row * N + i] * W[col * N + i];  // Wᵗ = W[col][i]
        }
        d_input[row * K + col] = sum;
    }
}

// d_W = inputᵗ × d_out
__global__ void matmul_backward_weight_kernel(const float* d_out, const float* input, float* d_W,
                                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // K
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += input[i * K + row] * d_out[i * N + col];  // inputᵗ × d_out
        }
        d_W[row * N + col] = sum;
    }
}
