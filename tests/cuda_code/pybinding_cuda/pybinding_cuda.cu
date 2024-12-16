#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

__global__ void addKernel(int* a, int* b, int* c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

void add_cuda(pybind11::array_t<int> a, pybind11::array_t<int> b, pybind11::array_t<int> c) {
    auto buf_a = a.request(), buf_b = b.request(), buf_c = c.request();
    int* d_a, *d_b, *d_c;

    int size = buf_a.shape[0];
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    cudaMemcpy(d_a, buf_a.ptr, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.ptr, size * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, size>>>(d_a, d_b, d_c);

    cudaMemcpy(buf_c.ptr, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

PYBIND11_MODULE(example, m) {
    m.def("add_cuda", &add_cuda, "A function that adds two arrays using CUDA");
}
