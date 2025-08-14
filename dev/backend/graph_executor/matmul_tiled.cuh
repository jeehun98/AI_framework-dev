#pragma once
#include <cuda_runtime.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 32   // 16도 가능하지만, Turing+에선 32 권장
#endif

// A[M,K] * B[K,N] = C[M,N]  (row-major)
void launch_matmul_tiled(const float* A, const float* B, float* C,
                         int M, int K, int N, cudaStream_t stream = 0);

// A[M,K] * B[K,N] + bias[1,N] (row-wise broadcasting)
void launch_matmul_bias_tiled(const float* A, const float* B, const float* bias, float* C,
                              int M, int K, int N, cudaStream_t stream = 0);
