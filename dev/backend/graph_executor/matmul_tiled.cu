#include "matmul_tiled.cuh"
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

// 표준 타일드 GEMM: 전치 없이 coalesced 로드
static __global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int K, int N) {
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // +1 패딩으로 bank conflict 완화
    __shared__ float As[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];

    float sum = 0.f;
    const int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        const int a_col = t * TILE_WIDTH + threadIdx.x; // A의 열 진행
        const int b_row = t * TILE_WIDTH + threadIdx.y; // B의 행 진행

        // A 로드 (coalesced: row 고정, col 증가)
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // B 로드 (coalesced: row 증가, col 고정)
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GEMM + bias row-wise add (에필로그에서 합치기)
static __global__ void matmul_bias_tiled_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                const float* __restrict__ bias, // [N]
                                                float* __restrict__ C,
                                                int M, int K, int N) {
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];

    float sum = 0.f;
    const int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        const int a_col = t * TILE_WIDTH + threadIdx.x;
        const int b_row = t * TILE_WIDTH + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col]; // row broadcasting
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C,
                         int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_bias_tiled(const float* A, const float* B, const float* bias, float* C,
                              int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_bias_tiled_kernel<<<grid, block, 0, stream>>>(A, B, bias, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}
