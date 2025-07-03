// activation.cu
#include <stdio.h>
#include <math.h>
#include "activation.cuh"

__global__ void activation_relu(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = val > 0 ? val : 0;
    }
}

__global__ void activation_sigmoid(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void activation_tanh(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = tanhf(val);
    }
}
