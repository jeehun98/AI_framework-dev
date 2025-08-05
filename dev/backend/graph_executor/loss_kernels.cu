// loss_kernels.cu (확장 버전): 비용 함수 + 각 손실 함수의 기울기 계산 포함

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// ================================
// MSE Forward (Loss)
// ================================
__global__ void mse_loss_kernel(const float* y_true, const float* y_pred, float* loss, int size) {
    __shared__ float cache[256];
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

__global__ void mse_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        grad_out[tid] = 2.0f * (y_pred[tid] - y_true[tid]) / size;  // 평균 포함
    }
}


// ================================
// Binary Cross-Entropy Forward
// ================================
__global__ void bce_loss_kernel(const float* y_true, const float* y_pred, float* loss, int size) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < size) {
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], 1e-7f), 1.0f - 1e-7f);
        temp += -yt * logf(yp) - (1.0f - yt) * logf(1.0f - yp);
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

__global__ void bce_loss_backward(
    const float* y_true,
    const float* y_pred,
    float* grad_out,
    int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        float yt = y_true[tid];
        float yp = y_pred[tid];  // sigmoid 후의 출력값
        float grad = (yp - yt) / size;
        grad_out[tid] = grad;

        if (tid < 4) {
            printf("[BCE_BACKWARD_FIXED] tid=%d: y_true=%.5f, y_pred=%.5f, grad=%.5f\n", tid, yt, yp, grad);
        }
    }
}



// ================================
// Categorical Cross-Entropy Forward
// ================================
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
            temp += -yt * logf(yp);
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

__global__ void cce_loss_backward(const float* y_true, const float* y_pred, float* grad_out, int batch_size, int num_classes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = batch_size * num_classes;
    if (tid < total) {
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], 1e-7f), 1.0f - 1e-7f);
        grad_out[tid] = -yt / yp / batch_size;  // 평균 포함한 softmax-CrossEntropy gradient
    }
}

// ========================
// MSE 손실 래퍼
// ========================
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

    return h_loss / size;
}

// ========================
// BCE 손실 래퍼
// ========================
float compute_bce_loss_cuda(float* y_true, float* y_pred, int size) {
    float* d_loss;
    float h_loss = 0.0f;

    // ✅ 입력값 디버그용 복사
    float debug_y_true[4];
    float debug_y_pred[4];
    cudaMemcpy(debug_y_true, y_true, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_y_pred, y_pred, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("[DEBUG BCE_LOSS] y_true[0:4] = %.5f %.5f %.5f %.5f\n",
        debug_y_true[0], debug_y_true[1], debug_y_true[2], debug_y_true[3]);

    printf("[DEBUG BCE_LOSS] y_pred[0:4] = %.5f %.5f %.5f %.5f\n",
        debug_y_pred[0], debug_y_pred[1], debug_y_pred[2], debug_y_pred[3]);

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

// ========================
// CCE 손실 래퍼
// ========================
float compute_cce_loss_cuda(float* y_true, float* y_pred, int batch_size, int num_classes) {
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int total = batch_size * num_classes;
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cce_loss_kernel<<<blocks, threads>>>(y_true, y_pred, d_loss, batch_size, num_classes);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / batch_size;
}
