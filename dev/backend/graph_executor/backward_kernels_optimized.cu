#include <cuda_runtime.h>
#include <math.h>
#include "logging_config.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// d_input = d_out * W^T
__global__ void matmul_backward_input_shared(const float* __restrict__ d_out,
                                             const float* __restrict__ W_T,
                                             float* __restrict__ d_input,
                                             int M, int N, int K) {
    __shared__ float d_out_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float W_T_tile  [TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // [0..M)
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // [0..K)

    float sum = 0.0f;

    const int phases = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < phases; ++ph) {
        int tiled_col = ph * TILE_WIDTH + threadIdx.x; // for d_out
        int tiled_row = ph * TILE_WIDTH + threadIdx.y; // for W_T

        d_out_tile[threadIdx.y][threadIdx.x] =
            (row < M && tiled_col < N) ? d_out[row * N + tiled_col] : 0.0f;

        // 주의: 공유메모리 bank conflict 줄이려고 [x][y]로 전치 로드
        W_T_tile[threadIdx.x][threadIdx.y] =
            (col < K && tiled_row < N) ? W_T[tiled_row * K + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += d_out_tile[threadIdx.y][k] * W_T_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        // 안정성 체크
        if (!isfinite(sum) || fabsf(sum) > 1e10f) {
            KPRINTF("[matmul_bw_input_shared][WARNING] row=%d col=%d sum=%.6f (clamped)\n", row, col, sum);
            sum = 0.0f;
        }
        d_input[row * K + col] = sum;

        // 한 번만 가벼운 요약 출력
        if (row == 0 && col == 0) {
            KPRINTF("[matmul_bw_input_shared] M=%d N=%d K=%d | d_input[0]=%.6f\n", M, N, K, sum);
            // 샘플 값(최대 4개)
            KPRINTF("[matmul_bw_input_shared] W_T[0..3]=%.6f %.6f %.6f %.6f\n",
                    W_T[0], (K>1?W_T[1]:0.f), (K>2?W_T[2]:0.f), (K>3?W_T[3]:0.f));
            KPRINTF("[matmul_bw_input_shared] d_out[0..3]=%.6f %.6f %.6f %.6f\n",
                    d_out[0], (N>1?d_out[1]:0.f), (N>2?d_out[2]:0.f), (N>3?d_out[3]:0.f));
        }
    }
}

// dW = X^T * dY  (input_T: [K x M], d_out: [M x N], d_weight: [K x N])
__global__ void matmul_backward_weight_shared(const float* __restrict__ input_T,
                                              const float* __restrict__ d_out,
                                              float* __restrict__ d_weight,
                                              int K, int N, int M) {
    __shared__ float input_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_out_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // [0..K)
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // [0..N)

    float sum = 0.0f;

    const int phases = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < phases; ++ph) {
        int tiled_col = ph * TILE_WIDTH + threadIdx.x;

        input_tile[threadIdx.y][threadIdx.x] =
            (row < K && tiled_col < M) ? input_T[row * M + tiled_col] : 0.0f;

        d_out_tile[threadIdx.y][threadIdx.x] =
            (tiled_col < M && col < N) ? d_out[tiled_col * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += input_tile[threadIdx.y][k] * d_out_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        if (!isfinite(sum) || fabsf(sum) > 1e10f) {
            KPRINTF("[matmul_bw_weight][WARNING] row=%d col=%d sum=%.6f (clamped)\n", row, col, sum);
            sum = 0.0f;
        }
        d_weight[row * N + col] = sum;

        if (row == 0 && col == 0) {
            KPRINTF("[matmul_bw_weight] K=%d N=%d M=%d | dW[0]=%.6f, input_T[0]=%.6f, d_out[0]=%.6f\n",
                    K, N, M, sum, input_T[0], d_out[0]);
        }
    }
}

// db = reduce_rows(dY)
__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0f;
    for (int i = 0; i < rows; ++i) sum += d_out[i * cols + col];

    if (!isfinite(sum) || fabsf(sum) > 1e10f) {
        KPRINTF("[add_backward_bias][WARNING] col=%d sum=%.6f (clamped)\n", col, sum);
        sum = 0.0f;
    }

    d_bias[col] = sum;

    if (col == 0) {
        KPRINTF("[add_backward_bias] rows=%d cols=%d | d_bias[0]=%.6f\n", rows, cols, sum);
    }
}

// dX = dY
__global__ void add_backward_input(const float* d_out, float* d_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float val = d_out[i];
    if (!isfinite(val) || fabsf(val) > 1e10f) {
        KPRINTF("[add_backward_input][WARNING] i=%d d_out=%.6f (clamped)\n", i, val);
        val = 0.0f;
    }

    d_input[i] = val;

    if (i == 0) {
        KPRINTF("[add_backward_input] size=%d | d_input[0]=%.6f (from d_out[0]=%.6f)\n", size, d_input[0], d_out[0]);
    }
}

__global__ void fill_gradient(float* grad, int total_size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_size) return;

    grad[i] = value;

    if (i == 0) {
        KPRINTF("[fill_gradient] total_size=%d value=%.6f | grad[0]=%.6f\n", total_size, value, grad[0]);
    }
}

// d_input = d_out * W^T  (simple version)
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
        sum += d_out[row * N + n] * W_T[n * K + col];
    }

    if (!isfinite(sum) || fabsf(sum) > 1e10f) {
        // 첫 원소에서만 경고
        if (idx == 0) {
            KPRINTF("[matmul_bw_input_simple][WARNING] bad grad: sum=%.6f row=%d col=%d\n", sum, row, col);
        }
        sum = 0.0f;
    }

    d_input[row * K + col] = sum;

    // 앞 몇 개만 요약 출력
    if (idx < 1) {
        KPRINTF("[matmul_bw_input_simple] M=%d N=%d K=%d | d_input[0]=%.6f\n", M, N, K, sum);
        KPRINTF("[matmul_bw_input_simple] W_T[0..3]=%.6f %.6f %.6f %.6f\n",
                W_T[0], (K>1?W_T[1]:0.f), (K>2?W_T[2]:0.f), (K>3?W_T[3]:0.f));
        KPRINTF("[matmul_bw_input_simple] d_out[0..3]=%.6f %.6f %.6f %.6f\n",
                d_out[0], (N>1?d_out[1]:0.f), (N>2?d_out[2]:0.f), (N>3?d_out[3]:0.f));
    }
}
