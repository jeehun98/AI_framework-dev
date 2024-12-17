#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// CUDA 커널 함수
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA 커널을 호출하는 함수
void add_vectors(float* a, float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Pybind11 래퍼 함수
PYBIND11_MODULE(kernel, m) {
    m.def("add_vectors", &add_vectors, "Add two vectors using CUDA");
}
