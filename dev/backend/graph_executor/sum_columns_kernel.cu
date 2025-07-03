#include <cuda_runtime.h>
#include "sum_columns_kernel.cuh"

// bias_gradient[col] += sum(grad_output[:, col])
__global__ void sum_columns_kernel(const float* grad_output, float* bias_gradient, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        sum += grad_output[row * cols + col];
    }

    bias_gradient[col] = sum;
}
