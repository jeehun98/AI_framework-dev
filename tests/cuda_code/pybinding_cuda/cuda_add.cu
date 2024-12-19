#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

__global__ void addKernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add(py::array_t<float> input1, py::array_t<float> input2, py::array_t<float> output) {
    auto buf1 = input1.request();
    auto buf2 = input2.request();
    auto buf_out = output.request();

    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);
    float* ptr_out = static_cast<float*>(buf_out.ptr);

    int size = buf1.size;

    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    cudaMemcpy(d_a, ptr1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, ptr2, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addKernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, size);

    cudaMemcpy(ptr_out, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

PYBIND11_MODULE(cuda_add, m) {
    m.def("vector_add", &vector_add, "Add two vectors using CUDA");
}
