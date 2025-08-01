// activation.cu
#include <stdio.h>
#include <math.h>
#include "activation.cuh"

__global__ void activation_relu(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);  // ✅ bias null-safe
        output[idx] = val > 0 ? val : 0;
    }
}

__device__ float safe_sigmoid(float x) {
    if (isnan(x)) return 0.5f;
    if (x < -30.0f) return 0.0f;
    if (x > 30.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));  // 정확성을 위해 expf 사용
}

__global__ void activation_sigmoid(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);
        output[idx] = safe_sigmoid(val);
    }
}

__global__ void activation_tanh(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + (bias ? bias[col] : 0.0f);  // ✅ bias null-safe
        float tanh_val = tanhf(val);
        output[idx] = isnan(tanh_val) ? 0.0f : tanh_val;
    }
}
