#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

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

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += d_out_tile[threadIdx.y][k] * W_T_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K) {
        // 오버플로우 체크 및 클리핑
        if (!isfinite(sum) || fabsf(sum) > 1e10f) {
            printf("[matmul_bw_input_shared][WARNING] row=%d col=%d sum=%.6f (clamped)\n", row, col, sum);
            sum = 0.0f;
        }

        d_input[row * K + col] = sum;

        // ✅ 디버그 출력: 처음 한 원소만 W_T, d_out 값 함께 출력
        if (row == 0 && col == 0) {
            printf("[matmul_backward_input_shared] M=%d N=%d K=%d\n", M, N, K);
            printf("[DEBUG] d_input[0] = %.6f\n", sum);

            printf("[DEBUG] W_T[0~3] = %.6f %.6f %.6f %.6f\n",
                   W_T[0], (K > 1 ? W_T[1] : 0.0f),
                   (K > 2 ? W_T[2] : 0.0f), (K > 3 ? W_T[3] : 0.0f));

            printf("[DEBUG] d_out[0~3] = %.6f %.6f %.6f %.6f\n",
                   d_out[0], (N > 1 ? d_out[1] : 0.0f),
                   (N > 2 ? d_out[2] : 0.0f), (N > 3 ? d_out[3] : 0.0f));
        }
    }
}


__global__ void matmul_backward_weight_shared(const float* __restrict__ input_T,  // [K x M]
                                              const float* __restrict__ d_out,    // [M x N]
                                              float* __restrict__ d_weight,       // [K x N]
                                              int K, int N, int M) {
    __shared__ float input_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_out_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // K
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;  // N

    float sum = 0.0f;

    for (int ph = 0; ph < (M + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int tiled_col = ph * TILE_WIDTH + threadIdx.x;

        input_tile[threadIdx.y][threadIdx.x] = (row < K && tiled_col < M)
            ? input_T[row * M + tiled_col] : 0.0f;

        d_out_tile[threadIdx.y][threadIdx.x] = (tiled_col < M && col < N)
            ? d_out[tiled_col * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += input_tile[threadIdx.y][k] * d_out_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < K && col < N) {
        d_weight[row * N + col] = sum;

        if (row == 0 && col == 0) {
            printf("[matmul_bw_weight] d_weight[0] = %f, input_T[0] = %f, d_out[0] = %f\n", sum, input_T[0], d_out[0]);
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

        if (col == 0) {
            printf("[add_backward_bias] d_bias[0] = %f\n", sum);
        }
    }
}

__global__ void add_backward_input(const float* d_out, float* d_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_input[i] = d_out[i];

        if (i == 0) {
            printf("[add_backward_input] d_input[0] = %f (from d_out[0] = %f)\n", d_input[i], d_out[i]);
        }
    }
}

__global__ void fill_gradient(float* grad, int total_size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        grad[i] = value;
        if (i == 0) {
            printf("[fill_gradient] grad[0] = %f\n", value);
        }
    }
}

__global__ void matmul_backward_input_simple(const float* __restrict__ d_out,
                                             const float* __restrict__ W_T,
                                             float* __restrict__ d_input,
                                             int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int row = idx / K;
    int col = idx % K;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dout = d_out[row * N + n];
        float wt = W_T[n * K + col];
        sum += dout * wt;
    }

    // 🔍 NaN 체크
    if (!isfinite(sum)) {
        printf("[matmul_bw_input_simple][NaN] sum=%.6f at row=%d, col=%d, d_out[0]=%.6f, W_T[0]=%.6f\n",
               sum, row, col, d_out[row * N], W_T[col]);
    }

    // 🔍 오버플로우 체크 및 클리핑
    if (!isfinite(sum) || fabsf(sum) > 1e10f) {
        if (idx == 0) {
            printf("[matmul_bw_input_simple][WARNING] bad grad: sum=%.6f, row=%d, col=%d\n", sum, row, col);
        }
        sum = 0.0f;
    }

    d_input[row * K + col] = sum;

    // ✅ 디버그 로그 추가: idx == 0일 때만 한 번 출력
    if (idx < 10) {
        printf("[matmul_bw_input_simple] M=%d, N=%d, K=%d\n", M, N, K);
        printf("[DEBUG] d_input[0] = %.6f\n", sum);

        printf("[DEBUG] W_T[0~3] = %.6f %.6f %.6f %.6f\n",
               W_T[0], (K > 1 ? W_T[1] : 0.0f),
               (K > 2 ? W_T[2] : 0.0f), (K > 3 ? W_T[3] : 0.0f));

        printf("[DEBUG] d_out[0~3] = %.6f %.6f %.6f %.6f\n",
               d_out[0], (N > 1 ? d_out[1] : 0.0f),
               (N > 2 ? d_out[2] : 0.0f), (N > 3 ? d_out[3] : 0.0f));
    }
}
