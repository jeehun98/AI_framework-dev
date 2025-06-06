#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16

// 전통적인 행렬 곱 (row x col, 열 방향 접근 비효율)
__global__ void matmul_shared_basic(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N; t += TILE_SIZE) {
        if (row < N && t + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// 최적화된 column access 기반 행렬 곱 (B 전치 없이 열을 행처럼 접근)
__global__ void matmul_shared_by_col_access(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N; t += TILE_SIZE) {
        if (row < N && t + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void fill_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand() % 10);
}

int main() {
    const int N = 64;
    const size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C1 = (float*)malloc(bytes);
    float *h_C2 = (float*)malloc(bytes);

    fill_matrix(h_A, N);
    fill_matrix(h_B, N);

    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 기본 행렬 곱
    cudaEventRecord(start);
    matmul_shared_basic<<<blocks, threads>>>(d_A, d_B, d_C1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_basic;
    cudaEventElapsedTime(&time_basic, start, stop);

    // 최적화된 전치 없는 방식
    cudaEventRecord(start);
    matmul_shared_by_col_access<<<blocks, threads>>>(d_A, d_B, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_optimized;
    cudaEventElapsedTime(&time_optimized, start, stop);

    cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C1[i] - h_C2[i]) > 1e-2f) ++errors;
    }

    printf("Basic Time:     %.3f ms\n", time_basic);
    printf("Optimized Time: %.3f ms\n", time_optimized);
    printf("Mismatch count: %d\n", errors);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_C1); free(h_C2);
    return 0;
}
