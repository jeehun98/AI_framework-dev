#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;
#define BLOCK_SIZE 256

float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

__global__ void mseLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = 0.0f;
    float sq = 0.0f;
    if (idx < n) {
        diff = y_pred[idx] - y_true[idx];

        // ✅ NaN 또는 inf 보호
        if (!isnan(diff) && !isinf(diff)) {
            sq = diff * diff;

            // ✅ overflow 방지 (상한 제한)
            if (sq > 1e10f) sq = 1e10f;
        }
        partial_sum[tid] = sq;
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


__global__ void mseGradKernel(const float* y_true, const float* y_pred, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] = 2.0f * (y_pred[idx] - y_true[idx]) / n;
}

float mse_loss(py::object y_true, py::object y_pred) {
    float* d_y_true = get_device_ptr(y_true);
    float* d_y_pred = get_device_ptr(y_pred);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1;
    for (auto s : shape) n *= s.cast<int>();

    float* d_loss;
    float h_loss = 0.0f;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mseLossKernel<<<gridSize, BLOCK_SIZE>>>(d_y_true, d_y_pred, d_loss, n);
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return h_loss / n;
}

void mse_grad(py::object y_true, py::object y_pred, py::object grad_out) {
    float* d_y_true = get_device_ptr(y_true);
    float* d_y_pred = get_device_ptr(y_pred);
    float* d_grad = get_device_ptr(grad_out);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1;
    for (auto s : shape) n *= s.cast<int>();

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mseGradKernel<<<gridSize, BLOCK_SIZE>>>(d_y_true, d_y_pred, d_grad, n);
}

PYBIND11_MODULE(losses_cuda, m) {
    m.def("mse_loss", &mse_loss, "CuPy 기반 MSE 손실 계산");
    m.def("mse_grad", &mse_grad, "CuPy 기반 MSE 손실 gradient 계산");
}
