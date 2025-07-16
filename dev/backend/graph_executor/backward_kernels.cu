#include <cuda_runtime.h>
#include "backward_kernels.cuh"

// ✅ dL/dx 계산 (input에 대한 gradient)
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

// ✅ dL/dW 계산 (weight에 대한 gradient)
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

// ✅ bias에 대한 gradient (broadcast된 경우 합산)
__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i)
            sum += d_out[i * cols + col];
        d_bias[col] = sum;
    }
}

// ✅ 입력에도 gradient 전파 (Add 레이어에서)
__global__ void add_backward_input(const float* d_out, float* d_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_input[i] = d_out[i];  // 그대로 복사
    }
}

// ✅ 최종 출력에 대한 gradient 수동 초기화용 커널
__global__ void fill_gradient(float* grad, int total_size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        grad[i] = value;
    }
}
