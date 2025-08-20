#pragma once
#include <cuda_runtime.h>

__global__ inline void ge_fill_kernel(float* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
