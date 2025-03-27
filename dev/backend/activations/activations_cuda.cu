#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// ReLU 활성화 함수 커널
__global__ void reluKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Sigmoid 활성화 함수 커널
__global__ void sigmoidKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

// Tanh 활성화 함수 커널
__global__ void tanhKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = tanhf(x[idx]);
    }
}

// CUDA를 활용한 활성화 함수 실행
void applyActivation(float* h_x, int n, void (*kernel)(float*, int)) {
    float* d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(d_x, n);

    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}

// Pybind를 이용한 Python에서 호출 가능한 래퍼 함수
py::array_t<float> apply_activation(py::array_t<float> input, std::string activation) {
    py::buffer_info buf = input.request();
    int n = buf.size;
    float* h_x = static_cast<float*>(buf.ptr);

    if (activation == "relu") {
        applyActivation(h_x, n, reluKernel);
    } else if (activation == "sigmoid") {
        applyActivation(h_x, n, sigmoidKernel);
    } else if (activation == "tanh") {
        applyActivation(h_x, n, tanhKernel);
    } else {
        throw std::invalid_argument("지원하지 않는 활성화 함수입니다. 'relu', 'sigmoid', 'tanh' 중 선택하세요.");
    }

    return py::array_t<float>(n, h_x);
}

// Pybind11 모듈 등록
PYBIND11_MODULE(activations_cuda, m) {
    m.def("apply_activation", &apply_activation, "CUDA 기반 활성화 함수 적용",
          py::arg("input"), py::arg("activation"));
}
