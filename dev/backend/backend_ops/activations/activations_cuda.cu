#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// -------------------------
// CuPy 포인터 추출
// -------------------------
float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

// -------------------------
// CUDA 커널 정의
// -------------------------
__global__ void reluKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void sigmoidKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

__global__ void tanhKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = tanhf(x[idx]);
}

__global__ void reluGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] = x[idx] > 0.0f ? grad[idx] : 0.0f;
}

__global__ void sigmoidGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = 1.0f / (1.0f + expf(-x[idx]));
        grad[idx] = grad[idx] * s * (1.0f - s);
    }
}

__global__ void tanhGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(x[idx]);
        grad[idx] = grad[idx] * (1.0f - t * t);
    }
}

// -------------------------
// CUDA Wrapper (CuPy 지원)
// -------------------------
void apply_activation(py::object x, std::string activation) {
    float* d_x = get_device_ptr(x);
    auto shape = x.attr("shape").cast<py::tuple>();
    int n = 1;
    for (auto s : shape) n *= s.cast<int>();

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (activation == "relu") {
        reluKernel<<<gridSize, blockSize>>>(d_x, n);
    } else if (activation == "sigmoid") {
        sigmoidKernel<<<gridSize, blockSize>>>(d_x, n);
    } else if (activation == "tanh") {
        tanhKernel<<<gridSize, blockSize>>>(d_x, n);
    } else {
        throw std::invalid_argument("지원하지 않는 활성화 함수입니다.");
    }
    cudaDeviceSynchronize();
}

void apply_activation_grad(py::object x, py::object grad, std::string activation) {
    float* d_x = get_device_ptr(x);
    float* d_grad = get_device_ptr(grad);
    auto shape = x.attr("shape").cast<py::tuple>();
    int n = 1;
    for (auto s : shape) n *= s.cast<int>();

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (activation == "relu") {
        reluGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else if (activation == "sigmoid") {
        sigmoidGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else if (activation == "tanh") {
        tanhGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else {
        throw std::invalid_argument("지원하지 않는 grad 활성화 함수입니다.");
    }
    cudaDeviceSynchronize();
}

// -------------------------
// Pybind11 모듈 등록
// -------------------------
PYBIND11_MODULE(activations_cuda, m) {
    m.def("apply_activation", &apply_activation, "CUDA 기반 활성화 함수 적용",
          py::arg("input"), py::arg("activation"));

    m.def("apply_activation_grad", &apply_activation_grad, "CUDA 기반 활성화 gradient 적용",
          py::arg("input"), py::arg("grad_input"), py::arg("activation"));
}
