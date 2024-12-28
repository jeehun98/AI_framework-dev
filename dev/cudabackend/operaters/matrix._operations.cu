#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>

namespace py = pybind11;

#define BLOCK_SIZE 16

// CUDA 커널: 행렬 덧셈
__global__ void addMatrices(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA 커널: 행렬 곱셈
__global__ void multiplyMatrices(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float value = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            value += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}

// Python 함수: 행렬 덧셈
py::array_t<float> matrix_add(py::array_t<float> a, py::array_t<float> b) {
    // 버퍼 생성
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.shape != b_info.shape) {
        throw std::runtime_error("Input matrices must have the same shape for addition!");
    }

    int rows = a_info.shape[0];
    int cols = a_info.shape[1];
    int size = rows * cols * sizeof(float);

    // CUDA 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트 데이터를 디바이스로 복사
    cudaMemcpy(d_A, a_info.ptr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b_info.ptr, size, cudaMemcpyHostToDevice);

    // CUDA 커널 설정
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CUDA 커널 실행
    addMatrices<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    // 결과 복사
    auto result = py::array_t<float>(a_info.size);
    py::buffer_info result_info = result.request();
    cudaMemcpy(result_info.ptr, d_C, size, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 결과 크기 설정
    result.resize({rows, cols});
    return result;
}

// Python 함수: 행렬 곱셈
py::array_t<float> matrix_multiply(py::array_t<float> a, py::array_t<float> b) {
    // 버퍼 생성
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.shape[1] != b_info.shape[0]) {
        throw std::runtime_error("Number of columns of A must match the number of rows of B for multiplication!");
    }

    int rowsA = a_info.shape[0];
    int colsA = a_info.shape[1];
    int rowsB = b_info.shape[0];
    int colsB = b_info.shape[1];
    int sizeC = rowsA * colsB * sizeof(float);

    // CUDA 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc(&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc(&d_C, sizeC);

    // 호스트 데이터를 디바이스로 복사
    cudaMemcpy(d_A, a_info.ptr, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b_info.ptr, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA 커널 설정
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CUDA 커널 실행
    multiplyMatrices<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    // 결과 복사
    auto result = py::array_t<float>(rowsA * colsB);
    py::buffer_info result_info = result.request();
    cudaMemcpy(result_info.ptr, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 결과 크기 설정
    result.resize({rowsA, colsB});
    return result;
}

// Pybind11 모듈 정의
PYBIND11_MODULE(matrix_operations, m) {
    m.doc() = "CUDA-based matrix addition and multiplication using Pybind11";
    m.def("matrix_add", &matrix_add, "Add two matrices using CUDA");
    m.def("matrix_multiply", &matrix_multiply, "Multiply two matrices using CUDA");
}
