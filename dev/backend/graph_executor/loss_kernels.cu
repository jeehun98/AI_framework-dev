// loss_kernels.cu (patched)

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err=(x); if (err!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); } } while(0)
#endif

// ================================
// MSE Forward (Loss)
// ================================
__global__ void mse_loss_kernel(const float* __restrict__ y_true,
                                const float* __restrict__ y_pred,
                                float* __restrict__ loss_sum,
                                int size) {
    extern __shared__ float cache[]; // size == blockDim.x
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    // grid-stride
    for (; tid < size; tid += blockDim.x * gridDim.x) {
        float d = y_true[tid] - y_pred[tid];
        tmp += d * d;
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    // block reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, cache[0]);
}

__global__ void mse_loss_backward(const float* __restrict__ y_true,
                                  const float* __restrict__ y_pred,
                                  float* __restrict__ grad_out,
                                  int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        grad_out[tid] = 2.0f * (y_pred[tid] - y_true[tid]) / (float)size; // 평균
    }
}

// ================================
// Binary Cross-Entropy Forward
// ================================
__global__ void bce_loss_kernel(const float* __restrict__ y_true,
                                const float* __restrict__ y_pred,
                                float* __restrict__ loss_sum,
                                int size) {
    extern __shared__ float cache[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    const float eps = 1e-7f;

    for (; tid < size; tid += blockDim.x * gridDim.x) {
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], eps), 1.0f - eps);
        tmp += -(yt * logf(yp) + (1.0f - yt) * logf(1.0f - yp));
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, cache[0]);
}

// Sigmoid 출력 a에 대한 dL/da
__global__ void bce_loss_backward(const float* __restrict__ y_true,
                                  const float* __restrict__ y_pred, // a
                                  float* __restrict__ grad_out,     // dL/da
                                  int size,
                                  int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    const float eps = 1e-7f;
    float a = fminf(fmaxf(y_pred[tid], eps), 1.f - eps);
    float y = y_true[tid];

    float denom = fmaxf(a * (1.f - a), eps);
    float dL_da = (a - y) / denom;

    float scale = (batch_size > 0) ? (1.f / (float)batch_size) : 1.f;
    grad_out[tid] = dL_da * scale;
}

// ================================
// Categorical Cross-Entropy
// ================================
__global__ void cce_loss_kernel(const float* __restrict__ y_true,
                                const float* __restrict__ y_pred,
                                float* __restrict__ loss_sum,
                                int batch_size, int num_classes) {
    extern __shared__ float cache[];
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    const float eps = 1e-7f;

    for (; b < batch_size; b += blockDim.x * gridDim.x) {
        const float* yt = y_true + b * num_classes;
        const float* yp = y_pred + b * num_classes;
        for (int j = 0; j < num_classes; ++j) {
            float yv = yt[j];
            float pv = fminf(fmaxf(yp[j], eps), 1.0f - eps);
            tmp += -yv * logf(pv);
        }
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, cache[0]);
}

__global__ void cce_loss_backward(const float* __restrict__ y_true,
                                  const float* __restrict__ y_pred,
                                  float* __restrict__ grad_out,
                                  int batch_size, int num_classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    if (tid < total) {
        const float eps = 1e-7f;
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], eps), 1.0f - eps);
        grad_out[tid] = -yt / yp / (float)batch_size;
    }
}

// ========================
// 래퍼 (평균 반환; 안전/오류 체크/디버그)
// ========================
static inline void launch_conf(int n, dim3& grid, dim3& block, size_t& shmem) {
    int threads = 256;
    int blocks  = std::min( std::max( (n + threads - 1) / threads, 1), 65535 );
    block = dim3(threads, 1, 1);
    grid  = dim3(blocks, 1, 1);
    shmem = threads * sizeof(float);
}

float compute_mse_loss_cuda(const float* y_true, const float* y_pred, int size) {
    if (size <= 0) return NAN;

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    dim3 grid, block; size_t shmem;
    launch_conf(size, grid, block, shmem);

    mse_loss_kernel<<<grid, block, shmem>>>(y_true, y_pred, d_sum, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum / (float)size;
}

float compute_bce_loss_cuda(const float* y_true, const float* y_pred, int size) {
    if (size <= 0) return NAN;

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    dim3 grid, block; size_t shmem;
    launch_conf(size, grid, block, shmem);

    bce_loss_kernel<<<grid, block, shmem>>>(y_true, y_pred, d_sum, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum / (float)size; // 평균
}

float compute_cce_loss_cuda(const float* y_true, const float* y_pred,
                            int batch_size, int num_classes) {
    if (batch_size <= 0 || num_classes <= 0) return NAN;

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    int work = batch_size; // per-batch reduce
    dim3 grid, block; size_t shmem;
    launch_conf(work, grid, block, shmem);

    cce_loss_kernel<<<grid, block, shmem>>>(y_true, y_pred, d_sum, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum / (float)batch_size;
}
