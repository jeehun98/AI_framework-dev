#include "add_kernel.cuh"

/**
 * @brief Element-wise Add with row broadcasting:
 *        output[i][j] = input[i][j] + bias[0][j]
 */
__global__ void add_kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}
