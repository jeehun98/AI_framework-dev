#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// MSE 손실 계산 커널
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

    // 병렬 reduction (block 내부 합산)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // 블록마다 하나의 partial sum 저장
    if (tid == 0) {
        atomicAdd(loss, partial_sum[0]);
    }
}

// Binary Cross Entropy 손실 계산 커널
__global__ void bceLossKernel(const float* y_true, const float* y_pred, float* loss, int n) {
    __shared__ float partial_sum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f - 1e-7f);  // 수치 안정성
        val = - (y * logf(p) + (1.0f - y) * logf(1.0f - p));
        partial_sum[tid] = val;
    } else {
        partial_sum[tid] = 0.0f;
    }

    __syncthreads();

    // 병렬 reduction
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

// 손실 함수 계산 실행 함수
float computeLoss(const float* h_y_true, const float* h_y_pred, int n, const std::string& loss_type) {
    float* d_y_true;
    float* d_y_pred;
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_y_pred, n * sizeof(float));
    cudaMalloc((void**)&d_loss, sizeof(float));
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (loss_type == "mse") {
        mseLossKernel<<<gridSize, blockSize>>>(d_y_true, d_y_pred, d_loss, n);
    } else if (loss_type == "bce") {
        bceLossKernel<<<gridSize, blockSize>>>(d_y_true, d_y_pred, d_loss, n);
    } else {
        throw std::invalid_argument("지원하지 않는 손실 함수입니다. 'mse', 'bce' 중 선택하세요.");
    }

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_loss);

    return h_loss / n;
}

// Pybind 래퍼
float compute_loss(py::array_t<float> y_true, py::array_t<float> y_pred, std::string loss_type) {
    py::buffer_info y_true_buf = y_true.request();
    py::buffer_info y_pred_buf = y_pred.request();

    if (y_true_buf.size != y_pred_buf.size) {
        throw std::invalid_argument("y_true와 y_pred의 크기가 일치하지 않습니다.");
    }

    float* h_y_true = static_cast<float*>(y_true_buf.ptr);
    float* h_y_pred = static_cast<float*>(y_pred_buf.ptr);
    int n = y_true_buf.size;

    return computeLoss(h_y_true, h_y_pred, n, loss_type);
}

// Pybind 모듈 정의
PYBIND11_MODULE(losses_cuda, m) {
    m.def("compute_loss", &compute_loss, "CUDA 기반 손실 함수 계산",
          py::arg("y_true"), py::arg("y_pred"), py::arg("loss_type"));
}
