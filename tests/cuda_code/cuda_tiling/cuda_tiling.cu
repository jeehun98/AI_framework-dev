#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16  // 타일 크기
#define N 1024  // 행렬 크기

__global__ void matrixMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0;

    for (int i = 0; i < n / TILE_SIZE; i++) {
        shared_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_SIZE + threadIdx.x];
        shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();  // 타일이 모두 로드될 때까지 대기

        for (int j = 0; j < TILE_SIZE; j++) {
            value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }
        __syncthreads();  // 현재 타일의 계산이 끝나면 다음 타일로 넘어가기 위해 대기
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

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

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 일부 결과 출력
    std::cout << "C[0]: " << h_C[0] << " C[N*N-1]: " << h_C[N * N - 1] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
