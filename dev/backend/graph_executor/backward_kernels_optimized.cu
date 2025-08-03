#include "backward_kernels_optimized.cuh"
#include <stdio.h>

__global__ void matmul_backward_input_shared(const float* __restrict__ d_out,
                                             const float* __restrict__ W_T,
                                             float* __restrict__ d_input,
                                             int M, int N, int K) {
    __shared__ float d_out_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float W_T_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int tiled_col = ph * TILE_WIDTH + threadIdx.x;
        int tiled_row = ph * TILE_WIDTH + threadIdx.y;

        if (row < M && tiled_col < N)
            d_out_tile[threadIdx.y][threadIdx.x] = d_out[row * N + tiled_col];
        else
            d_out_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && tiled_row < N)
            W_T_tile[threadIdx.x][threadIdx.y] = W_T[tiled_row * K + col];
        else
            W_T_tile[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += d_out_tile[threadIdx.y][k] * W_T_tile[threadIdx.x][k];

        __syncthreads();
    }

    if (row < M && col < K) {
        d_input[row * K + col] = sum;

        // ✅ 디버깅 출력
        if (row == 0 && col == 0 && (isnan(sum) || isinf(sum))) {
            printf("[matmul_backward_input] d_input[0] = %f (NaN/Inf) -> row=%d col=%d\n", sum, row, col);
        }
    }
}

__global__ void matmul_backward_weight_shared(const float* __restrict__ input,
                                              const float* __restrict__ d_out,
                                              float* __restrict__ d_weight,
                                              int M, int N, int K) {
    __shared__ float input_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_out_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // K
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;  // N

    float sum = 0.0f;

    for (int ph = 0; ph < (M + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int tiled_row = ph * TILE_WIDTH + threadIdx.y;
        int tiled_col = ph * TILE_WIDTH + threadIdx.x;

        if (tiled_row < M && row < K)
            input_tile[threadIdx.y][threadIdx.x] = input[tiled_row * K + row];
        else
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (tiled_row < M && col < N)
            d_out_tile[threadIdx.y][threadIdx.x] = d_out[tiled_row * N + col];
        else
            d_out_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += input_tile[k][threadIdx.y] * d_out_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < K && col < N) {
        d_weight[row * N + col] = sum;

        // ✅ 디버깅 출력
        if (row == 0 && col == 0 && (isnan(sum) || isinf(sum))) {
            printf("[matmul_backward_weight] d_weight[0] = %f (NaN/Inf) -> row=%d col=%d\n", sum, row, col);
        }
    }
}



// 그대로 유지 (Shared memory 없이도 효율적)
__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i)
            sum += d_out[i * cols + col];
        d_bias[col] = sum;
    }
}

__global__ void add_backward_input(const float* d_out, float* d_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_input[i] = d_out[i];
    }
}

__global__ void fill_gradient(float* grad, int total_size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        grad[i] = value;
    }
}
