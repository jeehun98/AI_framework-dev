#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void matrix_add_kernel(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        C[index] = A[index] + B[index];
    }
}

__global__ void matrix_mul_kernel(float* A, float* B, float* C, int rows, int cols, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float value = 0;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * cols + col];
        }
        C[row * cols + col] = value;

    }
}


void matrix_add(py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_c = c.request();

    float* A = static_cast<float*>(buf_a.ptr);
    float* B = static_cast<float*>(buf_b.ptr);
    float* C = static_cast<float*>(buf_c.ptr);

    int rows = buf_a.shape[0];
    int cols = buf_a.shape[1];

    float *d_A, *d_B, *d_C;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrix_mul(py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_c = c.request();

    float* A = static_cast<float*>(buf_a.ptr);
    float* B = static_cast<float*>(buf_b.ptr);
    float* C = static_cast<float*>(buf_c.ptr);

    int rows = buf_a.shape[0];
    int K = buf_a.shape[1];
    int cols = buf_b.shape[1];

    float *d_A, *d_B, *d_C;
    size_t size_A = rows * K * sizeof(float);
    size_t size_B = K * cols * sizeof(float);
    size_t size_C = rows * cols * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);  // ✅ C 초기화

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, K);

    // ✅ 커널 실행 완료 후 동기화
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


PYBIND11_MODULE(matrix_ops, m) {
    m.def("matrix_add", &matrix_add, "Matrix addition");
    m.def("matrix_mul", &matrix_mul, "Matrix multiplication");
}
