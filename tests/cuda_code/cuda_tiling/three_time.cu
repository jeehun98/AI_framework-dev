#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define N 512  // 행렬 크기

// GPU 비타일링(단순) 행렬 곱셈 커널
__global__ void matrixMulNaive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

// GPU 타일링을 사용한 행렬 곱셈 커널
__global__ void matrixMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0;

    for (int i = 0; i < n / TILE_SIZE; i++) {
        shared_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_SIZE + threadIdx.x];
        shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// CPU 행렬 곱셈 함수
void matrixMulCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float value = 0;
            for (int k = 0; k < n; k++) {
                value += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = value;
        }
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_naive = (float*)malloc(size);
    float *h_C_tiled = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);

    // 초기화
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // GPU 비타일링 행렬 곱셈
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);
    auto start_naive = std::chrono::high_resolution_clock::now();
    matrixMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> naive_duration = end_naive - start_naive;
    cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost);

    // GPU 타일링 행렬 곱셈
    auto start_tiled = std::chrono::high_resolution_clock::now();
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_tiled = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> tiled_duration = end_tiled - start_tiled;
    cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost);

    // CPU 행렬 곱셈
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

    // 성능 비교 출력
    std::cout << "Naive GPU matrix multiplication duration: " << naive_duration.count() << " ms" << std::endl;
    std::cout << "Tiled GPU matrix multiplication duration: " << tiled_duration.count() << " ms" << std::endl;
    std::cout << "CPU matrix multiplication duration: " << cpu_duration.count() << " ms" << std::endl;

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cpu);

    return 0;
}
