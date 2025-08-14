#pragma once
#include <cuda_runtime.h>

// out[row, col] = in[row, col] + bias[col]
void launch_add_bias_rowwise(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream = 0);
