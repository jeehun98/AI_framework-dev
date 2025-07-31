#include "loss_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// -----------------------------
// CUDA 커널: MSE
// -----------------------------
__global__ void mse_loss_kernel(const float* y_true, const float* y_pred, float* loss, int size) {
    __shared__ float cache[256];  // 블록 내 공유 메모리
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < size) {
        float diff = y_true[tid] - y_pred[tid];
        temp += diff * diff;
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    // 병렬 reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(loss, cache[0]);
}

float compute_mse_loss_cuda(float* y_true, float* y_pred, int size) {
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads>>>(y_true, y_pred, d_loss, size);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / size;  // 평균으로 정규화
}

// -----------------------------
// CUDA 커널: Binary Crossentropy
// -----------------------------
__global__ void bce_loss_kernel(const float* y_true, const float* y_pred, float* loss, int size) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < size) {
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], 1e-7f), 1.0f - 1e-7f);  // 안정성 보장
        temp += -yt * logf(yp) - (1 - yt) * logf(1 - yp);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(loss, cache[0]);
}

float compute_bce_loss_cuda(float* y_true, float* y_pred, int size) {
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    bce_loss_kernel<<<blocks, threads>>>(y_true, y_pred, d_loss, size);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / size;
}

// -----------------------------
// CUDA 커널: Categorical Crossentropy
// -----------------------------
__global__ void cce_loss_kernel(const float* y_true, const float* y_pred, float* loss, int batch_size, int num_classes) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < batch_size) {
        for (int j = 0; j < num_classes; ++j) {
            int idx = tid * num_classes + j;
            float yt = y_true[idx];
            float yp = fminf(fmaxf(y_pred[idx], 1e-7f), 1.0f - 1e-7f);
            temp += -yt * logf(yp);  // softmax 출력에 대해 cross-entropy 적용
        }
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(loss, cache[0]);
}

float compute_cce_loss_cuda(float* y_true, float* y_pred, int batch_size, int num_classes) {
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int total_elements = batch_size * num_classes;
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cce_loss_kernel<<<blocks, threads>>>(y_true, y_pred, d_loss, batch_size, num_classes);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / batch_size;
}
