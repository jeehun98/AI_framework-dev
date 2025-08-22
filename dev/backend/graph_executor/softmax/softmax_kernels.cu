// softmax_kernels.cu
#include "softmax_kernels.cuh"
#include <float.h>
#include <math.h>

#ifndef SOFTMAX_BLOCK_X
#define SOFTMAX_BLOCK_X 128
#endif

// softmax_kernels.cu
__global__ void softmax_forward_rowwise(
    const float* __restrict__ in, float* __restrict__ out,
    int rows, int cols, float temperature)
{
    int r = blockIdx.x;            // 한 블록이 한 행 처리
    if (r >= rows) return;

    // 1) row max
    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = in[r*cols + c] / temperature;
        m = fmaxf(m, v);
    }
    // warp/block reduce
    __shared__ float smax;
    // (간단 버전) thread 0 이 for-loop로 다시 스캔해도 OK
    if (threadIdx.x == 0) {
        m = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            float v = in[r*cols + c] / temperature;
            m = fmaxf(m, v);
        }
        smax = m;
    }
    __syncthreads();

    // 2) exp(x - max), sum
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float z = (in[r*cols + c] / temperature) - smax;
        float e = __expf(z);
        out[r*cols + c] = e;      // 임시 저장
        sum += e;
    }
    // (간단 버전) thread 0 이 다시 합산
    __shared__ float ssum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) s += out[r*cols + c];
        ssum = s;
    }
    __syncthreads();

    // 3) normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out[r*cols + c] = out[r*cols + c] / ssum;
    }
}

void launch_softmax_forward(const float* in, float* out,
                            int rows, int cols, float temperature,
                            cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(min(256, cols));
    softmax_forward_rowwise<<<grid, block, 0, stream>>>(in, out, rows, cols, temperature);
}

__global__ void softmax_backward_rowwise(
    const float* __restrict__ dY,
    const float* __restrict__ Y,
    float* __restrict__ dX,
    int rows, int cols, float temperature)
{
    int r = blockIdx.x;
    if (r >= rows) return;

    // row-wise dot(dY, Y)
    float dot = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dot += dY[r*cols + c] * Y[r*cols + c];
    }
    __shared__ float sdot;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) s += dY[r*cols + c] * Y[r*cols + c];
        sdot = s;
    }
    __syncthreads();

    // dX = (dY - dot*Y) * Y / T
    const float invT = 1.f / temperature;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float y = Y[r*cols + c];
        float gy = dY[r*cols + c];
        dX[r*cols + c] = (gy - sdot * y) * y * invT;
    }
}

void launch_softmax_backward(const float* grad_out, const float* out,
                             float* grad_in, int rows, int cols,
                             float temperature, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(min(256, cols));
    softmax_backward_rowwise<<<grid, block, 0, stream>>>(
        grad_out, out, grad_in, rows, cols, temperature);
}
