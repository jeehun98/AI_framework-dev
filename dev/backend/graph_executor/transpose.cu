// transpose.cu
#include <cuda_runtime.h>
#include "logging_config.h"   // KPRINTF 매크로

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// 타일 패딩(+1)로 shared memory bank conflict 완화
__global__ void transpose_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int rows, int cols) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    int in_r = blockIdx.y * TILE_WIDTH + threadIdx.y; // row
    int in_c = blockIdx.x * TILE_WIDTH + threadIdx.x; // col

    // load
    if (in_r < rows && in_c < cols) {
        tile[threadIdx.y][threadIdx.x] = input[in_r * cols + in_c];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // 전치된 좌표
    int out_r = blockIdx.x * TILE_WIDTH + threadIdx.y; // ← swap
    int out_c = blockIdx.y * TILE_WIDTH + threadIdx.x;

    if (out_r < cols && out_c < rows) {
        float v = tile[threadIdx.x][threadIdx.y];
        output[out_r * rows + out_c] = v;

        // 디버그: 첫 블록의 앞 몇 개만 인덱스 매핑 출력
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && threadIdx.x < 4) {
            int in_idx  = in_r  * cols + in_c;
            int out_idx = out_r * rows + out_c;
            KPRINTF("[transpose] input[%d] -> output[%d]\n", in_idx, out_idx);
        }
    }
}

inline void checkCuda(const char* where) {
#if DEBUG_KERNEL
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA][ERR] %s: %s\n", where, cudaGetErrorString(err));
    }
#endif
}

void launch_transpose(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((cols + TILE_WIDTH - 1) / TILE_WIDTH,
              (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    checkCuda("transpose_kernel launch");

#if DEBUG_KERNEL
    cudaDeviceSynchronize();
#endif
}
