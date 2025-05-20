#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

__global__ void mseLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[256];
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

    if (tid == 0) {
        atomicAdd(loss, partial_sum[0]);
    }
}

__global__ void bceLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f - 1e-7f);
        val = - (y * logf(p) + (1.0f - y) * logf(1.0f - p));
        partial_sum[tid] = val;
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

    if (tid == 0) {
        atomicAdd(loss, partial_sum[0]);
    }
}

// 공통 CUDA 실행 함수
float launchLossKernel(const float* h_y_true, const float* h_y_pred, int n, void(*kernel)(const float*, const float*, float*, int)) {
    float* d_y_true;
    float* d_y_pred;
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_y_true, n * sizeof(float));
    cudaMalloc(&d_y_pred, n * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    kernel<<<gridSize, blockSize>>>(d_y_true, d_y_pred, d_loss, n);

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_loss);

    return h_loss / n;
}

// ==========================
//      Pybind 함수 정의
// ==========================

float mse_loss(py::array_t<float> y_true, py::array_t<float> y_pred) {
    auto y_true_buf = y_true.request();
    auto y_pred_buf = y_pred.request();

    if (y_true_buf.size != y_pred_buf.size) {
        throw std::invalid_argument("y_true와 y_pred의 크기가 일치하지 않습니다.");
    }

    return launchLossKernel(
        static_cast<float*>(y_true_buf.ptr),
        static_cast<float*>(y_pred_buf.ptr),
        y_true_buf.size,
        mseLossKernel
    );
}

float binary_crossentropy(py::array_t<float> y_true, py::array_t<float> y_pred) {
    auto y_true_buf = y_true.request();
    auto y_pred_buf = y_pred.request();

    if (y_true_buf.size != y_pred_buf.size) {
        throw std::invalid_argument("y_true와 y_pred의 크기가 일치하지 않습니다.");
    }

    return launchLossKernel(
        static_cast<float*>(y_true_buf.ptr),
        static_cast<float*>(y_pred_buf.ptr),
        y_true_buf.size,
        bceLossKernel
    );
}

// Placeholder for Categorical Crossentropy (추후 구현 예정)
float categorical_crossentropy(py::array_t<float> y_true, py::array_t<float> y_pred) {
    throw std::runtime_error("Categorical Crossentropy는 아직 구현되지 않았습니다.");
}

// ==========================
//        모듈 등록
// ==========================
PYBIND11_MODULE(losses_cuda, m) {
    m.def("mse_loss", &mse_loss, "MSE 손실 계산");
    m.def("binary_crossentropy", &binary_crossentropy, "Binary Crossentropy 계산");
    m.def("categorical_crossentropy", &categorical_crossentropy, "Categorical Crossentropy 계산");
}
