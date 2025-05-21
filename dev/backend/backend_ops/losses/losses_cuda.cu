// losses_cuda.cu
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;
#define BLOCK_SIZE 256

// --------------------------------------
// Loss Kernels
// --------------------------------------
__global__ void mseLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = 0.0f;
    if (idx < n) {
        diff = y_pred[idx] - y_true[idx];
        partial_sum[tid] = diff * diff;
    } else {
        partial_sum[tid] = 0.0f;
    }

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss, partial_sum[0]);
}

__global__ void bceLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f - 1e-7f);
        partial_sum[tid] = - (y * logf(p) + (1 - y) * logf(1 - p));
    } else {
        partial_sum[tid] = 0.0f;
    }

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sum[tid] += partial_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss, partial_sum[0]);
}

__global__ void cceLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f);
        partial_sum[tid] = - y_true[idx] * logf(p);
    } else {
        partial_sum[tid] = 0.0f;
    }

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sum[tid] += partial_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss, partial_sum[0]);
}

// --------------------------------------
// Gradient Kernels
// --------------------------------------
__global__ void mseGradKernel(const float* y_true, const float* y_pred, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] = 2.0f * (y_pred[idx] - y_true[idx]) / n;
}

__global__ void bceGradKernel(const float* y_true, const float* y_pred, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f - 1e-7f);
        grad[idx] = (p - y) / (p * (1.0f - p) * n);
    }
}

__global__ void cceGradKernel(const float* y_true, const float* y_pred, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f);
        grad[idx] = - y_true[idx] / p / n;
    }
}

// --------------------------------------
// 실행 함수
// --------------------------------------
float launchLossKernel(const float* h_y_true, const float* h_y_pred, int n, void(*kernel)(const float*, const float*, float*, int)) {
    float *d_y_true, *d_y_pred, *d_loss;
    float h_loss = 0.0f;
    cudaMalloc(&d_y_true, n * sizeof(float));
    cudaMalloc(&d_y_pred, n * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel<<<gridSize, BLOCK_SIZE>>>(d_y_true, d_y_pred, d_loss, n);

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_y_true); cudaFree(d_y_pred); cudaFree(d_loss);
    return h_loss / n;
}

py::array_t<float> launchGradKernel(py::array_t<float> y_true, py::array_t<float> y_pred, void(*kernel)(const float*, const float*, float*, int)) {
    auto buf_true = y_true.request();
    auto buf_pred = y_pred.request();
    int n = buf_true.size;
    float *h_y_true = static_cast<float*>(buf_true.ptr);
    float *h_y_pred = static_cast<float*>(buf_pred.ptr);
    py::array_t<float> grad(n);
    float* h_grad = static_cast<float*>(grad.request().ptr);

    float *d_y_true, *d_y_pred, *d_grad;
    cudaMalloc(&d_y_true, n * sizeof(float));
    cudaMalloc(&d_y_pred, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));

    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, n * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel<<<gridSize, BLOCK_SIZE>>>(d_y_true, d_y_pred, d_grad, n);

    cudaMemcpy(h_grad, d_grad, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_y_true); cudaFree(d_y_pred); cudaFree(d_grad);

    return grad;
}

// --------------------------------------
// Pybind 등록
// --------------------------------------
PYBIND11_MODULE(losses_cuda, m) {
    m.def("mse_loss", [](py::array_t<float> y, py::array_t<float> p) {
        return launchLossKernel((float*)y.request().ptr, (float*)p.request().ptr, y.size(), mseLossKernel);
    });
    m.def("mse_grad", [](py::array_t<float> y, py::array_t<float> p) {
        return launchGradKernel(y, p, mseGradKernel);
    });
    m.def("binary_crossentropy", [](py::array_t<float> y, py::array_t<float> p) {
        return launchLossKernel((float*)y.request().ptr, (float*)p.request().ptr, y.size(), bceLossKernel);
    });
    m.def("bce_grad", [](py::array_t<float> y, py::array_t<float> p) {
        return launchGradKernel(y, p, bceGradKernel);
    });
    m.def("categorical_crossentropy", [](py::array_t<float> y, py::array_t<float> p) {
        return launchLossKernel((float*)y.request().ptr, (float*)p.request().ptr, y.size(), cceLossKernel);
    });
    m.def("cce_grad", [](py::array_t<float> y, py::array_t<float> p) {
        return launchGradKernel(y, p, cceGradKernel);
    });
}
