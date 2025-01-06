#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 16 // 행렬 곱에서의 타일 크기

// 행렬 합 커널
__global__ void matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// 행렬 곱 커널
__global__ void matrix_mul(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; i++) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// 행렬 합 함수
void add_matrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int rows, int cols) {
    int size = rows * cols;
    float *d_A, *d_B, *d_C;

    // GPU 메모리 할당
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    // 데이터 복사 (Host to Device)
    cudaMemcpy(d_A, A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // 블록 및 그리드 크기 설정
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + 15) / 16, (rows + 15) / 16);

    // 커널 호출
    matrix_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    // 결과 복사 (Device to Host)
    cudaMemcpy(C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 행렬 곱 함수
void multiply_matrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                       int rowsA, int colsA, int colsB) {
    int sizeA = rowsA * colsA;
    int sizeB = colsA * colsB;
    int sizeC = rowsA * colsB;

    float *d_A, *d_B, *d_C;

    // GPU 메모리 할당
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // 데이터 복사 (Host to Device)
    cudaMemcpy(d_A, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // 블록 및 그리드 크기 설정
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((colsB + TILE_SIZE - 1) / TILE_SIZE, (rowsA + TILE_SIZE - 1) / TILE_SIZE);

    // 커널 호출
    matrix_mul<<<gridSize, blockSize>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // 결과 복사 (Device to Host)
    cudaMemcpy(C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
