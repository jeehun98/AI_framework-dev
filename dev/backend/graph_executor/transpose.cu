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

        // 디버깅 출력: 앞부분 몇 개만 확인
        if (row == 0 && col < 10) {
            printf("[transpose] input[%d] = %.6f -> output[%d] = %.6f\n",
                   row * cols + col,
                   val,
                   col * rows + row,
                   output[col * rows + row]);  // 이 값은 실제로는 device memory에 있기 때문에 의미 없음
        }
    }
}

void launch_transpose(const float* input, float* output, int rows, int cols) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);
    transpose_kernel<<<gridDim, blockDim>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
