// softmax_kernels.cuh
#pragma once
#include <cuda_runtime.h>

void launch_softmax_forward(const float* in, float* out,
                            int rows, int cols, float temperature,
                            cudaStream_t stream);
void launch_softmax_backward(const float* grad_out, const float* out,
                             float* grad_in, int rows, int cols,
                             float temperature, cudaStream_t stream);
