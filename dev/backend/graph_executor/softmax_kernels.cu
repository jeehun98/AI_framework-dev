// softmax_kernels.cu
#include "softmax_kernels.cuh"
#include <float.h>
#include <math.h>

#ifndef SOFTMAX_BLOCK_X
#define SOFTMAX_BLOCK_X 128
#endif

// 각 block이 한 row 처리 (cols가 크면 루프)
__global__ void softmax_forward_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int rows, int cols, float inv_temp)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // 1) row max
    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = in[row * cols + c] * inv_temp;
        m = fmaxf(m, v);
    }
    // warp/block reduce
    __shared__ float smax;
    // 간단 공유메모리 reduce
    __shared__ float buf[SOFTMAX_BLOCK_X];
    buf[threadIdx.x] = m;
    __syncthreads();
    // reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) smax = buf[0];
    __syncthreads();

    // 2) exp(x-m) & sum
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = in[row * cols + c] * inv_temp;
        float e = expf(v - smax);
        out[row * cols + c] = e; // 임시 저장
        sum += e;
    }
    // sum reduce
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float rsum = buf[0] + 1e-12f; // 안정성

    // 3) normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out[row * cols + c] = out[row * cols + c] / rsum;
    }
}

__global__ void softmax_backward_kernel(const float* __restrict__ grad_out,
                                        const float* __restrict__ out,
                                        float* __restrict__ grad_in,
                                        int rows, int cols, float inv_temp)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    // s = sum_j (dY_j * Y_j)
    float s = 0.f;
    __shared__ float buf[SOFTMAX_BLOCK_X];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float gy = grad_out[row * cols + c];
        float y  = out[row * cols + c];
        s += gy * y;
    }
    buf[threadIdx.x] = s;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (threadIdx.x < r) buf[threadIdx.x] += buf[threadIdx.x + r];
        __syncthreads();
    }
    float S = buf[0];

    // dX = (Y ⊙ (dY - S)) / τ
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float y  = out[row * cols + c];
        float gy = grad_out[row * cols + c];
        grad_in[row * cols + c] = (y * (gy - S)) * inv_temp;
    }
}

void launch_softmax_forward(const float* in, float* out,
                            int rows, int cols, float temperature,
                            cudaStream_t stream)
{
    int blocks = rows;
    int threads = SOFTMAX_BLOCK_X;
    float inv_temp = 1.0f / fmaxf(temperature, 1e-6f);
    softmax_forward_kernel<<<blocks, threads, 0, stream>>>(in, out, rows, cols, inv_temp);
}

void launch_softmax_backward(const float* grad_out, const float* out,
                             float* grad_in, int rows, int cols,
                             float temperature, cudaStream_t stream)
{
    int blocks = rows;
    int threads = SOFTMAX_BLOCK_X;
    float inv_temp = 1.0f / fmaxf(temperature, 1e-6f);
    softmax_backward_kernel<<<blocks, threads, 0, stream>>>(grad_out, out, grad_in, rows, cols, inv_temp);
}
