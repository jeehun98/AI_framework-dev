// activation.cu
#include <stdio.h>
#include <math.h>
#include "activation.cuh"

__global__ void activation_relu(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);
        output[idx] = val > 0 ? val : 0;
    }
}

__device__ float safe_sigmoid(float x) {
    if (isnan(x)) return 0.5f;
    // 완전 0/1이 아니라 살짝 여유(ε) 남김
    const float eps = 1e-7f;
    if (x < -30.0f) return eps;
    if (x >  30.0f) return 1.0f - eps;
    return 1.0f / (1.0f + expf(-x));
}

__global__ void activation_sigmoid(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);
        float s = safe_sigmoid(val);
        // 안전망: 0~1 범위 보장
        output[idx] = fminf(fmaxf(s, 0.0f), 1.0f);
    }
}

__global__ void activation_tanh(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);
        float t = tanhf(val);
        output[idx] = isnan(t) ? 0.0f : t;
    }
}
