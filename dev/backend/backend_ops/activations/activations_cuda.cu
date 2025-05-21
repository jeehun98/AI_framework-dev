#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// -------------------------
// 1. CUDA 커널 정의
// -------------------------

// ✅ ReLU Forward
__global__ void reluKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// ✅ Sigmoid Forward
__global__ void sigmoidKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

// ✅ Tanh Forward
__global__ void tanhKernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = tanhf(x[idx]);
    }
}

// ✅ ReLU Backward (grad_output *= x > 0)
__global__ void reluGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = x[idx] > 0.0f ? grad[idx] : 0.0f;
    }
}

// ✅ Sigmoid Backward (grad_output *= s * (1 - s))
__global__ void sigmoidGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = 1.0f / (1.0f + expf(-x[idx]));
        grad[idx] = grad[idx] * s * (1.0f - s);
    }
}

// ✅ Tanh Backward (grad_output *= 1 - tanh^2)
__global__ void tanhGradKernel(float* x, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(x[idx]);
        grad[idx] = grad[idx] * (1.0f - t * t);
    }
}

// -------------------------
// 2. Forward CUDA Wrapper
// -------------------------

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

// -------------------------
// 3. Backward CUDA Wrapper
// -------------------------

void applyActivationGrad(float* h_x, float* h_grad, int n, const std::string& activation) {
    float *d_x, *d_grad;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_grad, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (activation == "relu") {
        reluGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else if (activation == "sigmoid") {
        sigmoidGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else if (activation == "tanh") {
        tanhGradKernel<<<gridSize, blockSize>>>(d_x, d_grad, n);
    } else {
        cudaFree(d_x);
        cudaFree(d_grad);
        throw std::invalid_argument("지원하지 않는 grad 활성화 함수입니다.");
    }

    cudaMemcpy(h_grad, d_grad, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_grad);
}

// -------------------------
// 4. Pybind11 함수 정의
// -------------------------

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
        throw std::invalid_argument("지원하지 않는 활성화 함수입니다.");
    }

    return input;
}

py::array_t<float> apply_activation_grad(py::array_t<float> input, py::array_t<float> grad_input, std::string activation) {
    py::buffer_info buf_x = input.request();
    py::buffer_info buf_grad = grad_input.request();

    if (buf_x.size != buf_grad.size) {
        throw std::invalid_argument("입력과 grad_input 크기가 일치하지 않습니다.");
    }

    float* h_x = static_cast<float*>(buf_x.ptr);
    float* h_grad = static_cast<float*>(buf_grad.ptr);
    int n = buf_x.size;

    applyActivationGrad(h_x, h_grad, n, activation);

    return grad_input;
}

// -------------------------
// 5. Pybind11 모듈 등록
// -------------------------

PYBIND11_MODULE(activations_cuda, m) {
    m.def("apply_activation", &apply_activation, "CUDA 기반 활성화 함수 적용",
          py::arg("input"), py::arg("activation"));

    m.def("apply_activation_grad", &apply_activation_grad, "CUDA 기반 활성화 함수 gradient 적용",
          py::arg("input"), py::arg("grad_input"), py::arg("activation"));
}
