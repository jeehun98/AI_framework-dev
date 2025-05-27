#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

__global__ void matmul_shared_basic(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; ++t) {
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }
    C[row * N + col] = sum;
}

__global__ void matmul_shared_transpose(const float* A, const float* B_T, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B_T[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; ++t) {
        int tiled_col = t * TILE_SIZE + threadIdx.x;
        int tiled_row = t * TILE_SIZE + threadIdx.y;

        // A is accessed row-wise
        tile_A[threadIdx.y][threadIdx.x] =
            A[row * N + tiled_col];

        // B_T[col * N + k] == B[k][col]
        tile_B_T[threadIdx.y][threadIdx.x] =
            B_T[col * N + tiled_row];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B_T[threadIdx.x][k];

        __syncthreads();
    }

    C[row * N + col] = sum;
}


void transpose(const float* B, float* B_T, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B_T[j * N + i] = B[i * N + j];
}

void fill_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand() % 100) / 100.0f;
}

int main() {
    const int N = 64;
    const size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_B_T = (float*)malloc(bytes);
    float *h_C1 = (float*)malloc(bytes);
    float *h_C2 = (float*)malloc(bytes);

    fill_matrix(h_A, N);
    fill_matrix(h_B, N);
    transpose(h_B, h_B_T, N);

    float *d_A, *d_B, *d_B_T, *d_C1, *d_C2;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_B_T, bytes);
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_T, h_B_T, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Shared memory - basic
    cudaEventRecord(start);
    matmul_shared_basic<<<blocks, threads>>>(d_A, d_B, d_C1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared_basic;
    cudaEventElapsedTime(&ms_shared_basic, start, stop);

    // Shared memory - transpose
    cudaEventRecord(start);
    matmul_shared_transpose<<<blocks, threads>>>(d_A, d_B_T, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared_trans;
    cudaEventElapsedTime(&ms_shared_trans, start, stop);

    printf("Shared Basic Time:     %.3f ms\n", ms_shared_basic);
    printf("Shared Transpose Time: %.3f ms\n", ms_shared_trans);

    cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C1[i] - h_C2[i]) > 1e-2f)
            ++errors;
    }
    printf("Mismatch count: %d\n", errors);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_B_T); cudaFree(d_C1); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_B_T); free(h_C1); free(h_C2);
    return 0;
}
