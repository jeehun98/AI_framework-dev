// matmul_shared_optimized.cuh
#ifndef MATMUL_SHARED_OPTIMIZED_CUH
#define MATMUL_SHARED_OPTIMIZED_CUH

#define TILE_WIDTH 16

__global__ void matmul_shared_kernel_coalesced(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int A_rows, int A_cols, int B_cols);

#endif  // MATMUL_SHARED_OPTIMIZED_CUH
