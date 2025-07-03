// matmul_shared_optimized_kernel.cu
#include "matmul_shared_optimized.cuh"

__global__ void matmul_shared_kernel_coalesced(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int A_rows, int A_cols, int B_cols) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];  // B는 transpose 형태로 저장

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int tiled_col = ph * TILE_WIDTH + threadIdx.x;
        int tiled_row = ph * TILE_WIDTH + threadIdx.y;

        // A 로드: coalesced (row 고정, col 증가)
        if (row < A_rows && tiled_col < A_cols)
            A_tile[threadIdx.y][threadIdx.x] = A[row * A_cols + tiled_col];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // B 로드: transposed 형태로 shared memory에 저장 (row 증가, col 고정 → transpose)
        if (col < B_cols && tiled_row < A_cols)
            B_tile[threadIdx.x][threadIdx.y] = B[tiled_row * B_cols + col];
        else
            B_tile[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += A_tile[threadIdx.y][k] * B_tile[threadIdx.x][k];  // Transposed 형태 사용

        __syncthreads();
    }

    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = sum;
}
