// transpose.cu

#include <cuda_runtime.h>
#include <iostream>
#include "transpose.cuh"

#define TILE_WIDTH 16

__global__ void transpose_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < rows && col < cols) {
        float val = input[row * cols + col];
        output[col * rows + row] = val;

        if (row == 0 && col == 0) {
            // printf("[transpose] input[0] = %f, output[0] = %f\n", input[0], output[0]);
        }

        // 임의로 한 셀 더 출력
        if (row == 1 && col == 0) {
            // printf("[transpose] input[cols] = %f, output[rows] = %f\n", input[cols], output[rows]);
        }
    }
}


void launch_transpose(const float* input, float* output, int rows, int cols) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    transpose_kernel<<<gridDim, blockDim>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
