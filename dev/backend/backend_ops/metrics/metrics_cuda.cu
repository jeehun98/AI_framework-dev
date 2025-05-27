#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// GPU 포인터 추출 유틸리티
float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

int* get_device_ptr_int(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<int*>(ptr);
}

__global__ void accuracyKernel(const int* y_true, const int* y_pred, int* correct, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && y_true[idx] == y_pred[idx]) {
        atomicAdd(correct, 1);
    }
}

__global__ void precisionKernel(const int* y_true, const int* y_pred, int* true_positive, int* false_positive, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && y_pred[idx] == 1) {
        if (y_true[idx] == 1) atomicAdd(true_positive, 1);
        else atomicAdd(false_positive, 1);
    }
}

__global__ void recallKernel(const int* y_true, const int* y_pred, int* true_positive, int* false_negative, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && y_true[idx] == 1) {
        if (y_pred[idx] == 1) atomicAdd(true_positive, 1);
        else atomicAdd(false_negative, 1);
    }
}

__global__ void mseKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float diff = y_true[idx] - y_pred[idx];
        // ✅ 수치 안정성 강화
        if (!isfinite(diff) || fabsf(diff) > 1e3f) {
            diff = 0.0f;
        }
        val = diff * diff;
    }
    partial[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss, partial[0]);
}


float accuracy(py::object y_true, py::object y_pred) {
    int* d_y_true = get_device_ptr_int(y_true);
    int* d_y_pred = get_device_ptr_int(y_pred);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1; for (auto s : shape) n *= s.cast<int>();

    int* d_correct;
    int h_correct = 0;
    cudaMalloc(&d_correct, sizeof(int));
    cudaMemset(d_correct, 0, sizeof(int));
    accuracyKernel<<<(n + 255) / 256, 256>>>(d_y_true, d_y_pred, d_correct, n);
    cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_correct);
    return static_cast<float>(h_correct) / n;
}

float precision(py::object y_true, py::object y_pred) {
    int *d_y_true = get_device_ptr_int(y_true);
    int *d_y_pred = get_device_ptr_int(y_pred);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1; for (auto s : shape) n *= s.cast<int>();

    int *tp, *fp;
    int h_tp = 0, h_fp = 0;
    cudaMalloc(&tp, sizeof(int)); cudaMemset(tp, 0, sizeof(int));
    cudaMalloc(&fp, sizeof(int)); cudaMemset(fp, 0, sizeof(int));
    precisionKernel<<<(n + 255) / 256, 256>>>(d_y_true, d_y_pred, tp, fp, n);
    cudaMemcpy(&h_tp, tp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_fp, fp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(tp); cudaFree(fp);
    return h_tp + h_fp == 0 ? 0.0f : static_cast<float>(h_tp) / (h_tp + h_fp);
}

float recall(py::object y_true, py::object y_pred) {
    int *d_y_true = get_device_ptr_int(y_true);
    int *d_y_pred = get_device_ptr_int(y_pred);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1; for (auto s : shape) n *= s.cast<int>();

    int *tp, *fn;
    int h_tp = 0, h_fn = 0;
    cudaMalloc(&tp, sizeof(int)); cudaMemset(tp, 0, sizeof(int));
    cudaMalloc(&fn, sizeof(int)); cudaMemset(fn, 0, sizeof(int));
    recallKernel<<<(n + 255) / 256, 256>>>(d_y_true, d_y_pred, tp, fn, n);
    cudaMemcpy(&h_tp, tp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_fn, fn, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(tp); cudaFree(fn);
    return h_tp + h_fn == 0 ? 0.0f : static_cast<float>(h_tp) / (h_tp + h_fn);
}

float f1_score(py::object y_true, py::object y_pred) {
    float p = precision(y_true, y_pred);
    float r = recall(y_true, y_pred);
    return (p + r) == 0.0f ? 0.0f : 2.0f * p * r / (p + r);
}

float mse(py::object y_true, py::object y_pred) {
    float* d_y_true = get_device_ptr(y_true);
    float* d_y_pred = get_device_ptr(y_pred);
    auto shape = y_true.attr("shape").cast<py::tuple>();
    int n = 1; for (auto s : shape) n *= s.cast<int>();

    float* d_loss;
    float h_loss = 0.0f;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));
    mseKernel<<<(n + 255) / 256, 256>>>(d_y_true, d_y_pred, d_loss, n);
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return h_loss / n;
}

PYBIND11_MODULE(metrics_cuda, m) {
    m.def("accuracy", &accuracy);
    m.def("precision", &precision);
    m.def("recall", &recall);
    m.def("f1_score", &f1_score);
    m.def("mse", &mse);
}
