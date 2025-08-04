// === backward_kernels_optimized.cu ===

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 16

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

        d_out_tile[threadIdx.y][threadIdx.x] = (row < M && tiled_col < N)
            ? d_out[row * N + tiled_col] : 0.0f;

        W_T_tile[threadIdx.x][threadIdx.y] = (col < K && tiled_row < N)
            ? W_T[tiled_row * K + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a = d_out_tile[threadIdx.y][k];
            float b = W_T_tile[threadIdx.x][k];
            if (isnan(a) || isnan(b) || isinf(a) || isinf(b)) {
                if (row == 0 && col == 0) {
                    printf("[matmul_backward_input][NaN] tile input NaN/Inf at k=%d \u2192 a=%f, b=%f\n", k, a, b);
                }
                continue;
            }
            float prod = a * b;
            if (!isnan(prod) && !isinf(prod)) {
                sum += prod;
            }
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        d_input[row * K + col] = sum;

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

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (M + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int tiled_row = ph * TILE_WIDTH + threadIdx.y;

        input_tile[threadIdx.y][threadIdx.x] = (tiled_row < M && row < K)
            ? input[tiled_row * K + row] : 0.0f;

        d_out_tile[threadIdx.y][threadIdx.x] = (tiled_row < M && col < N)
            ? d_out[tiled_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a = input_tile[k][threadIdx.y];
            float b = d_out_tile[k][threadIdx.x];
            if (isnan(a) || isnan(b) || isinf(a) || isinf(b)) {
                if (row == 0 && col == 0) {
                    printf("[matmul_backward_weight][NaN] a=%f, b=%f, k=%d\n", a, b, k);
                }
                continue;
            }
            float prod = a * b;
            if (!isnan(prod) && !isinf(prod)) {
                sum += prod;
            }
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        d_weight[row * N + col] = sum;

        if (row == 0 && col == 0 && (isnan(sum) || isinf(sum))) {
            printf("[matmul_backward_weight] d_weight[0] = %f (NaN/Inf) -> row=%d col=%d\n", sum, row, col);
        }
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
