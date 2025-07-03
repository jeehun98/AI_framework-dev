#pragma once

__global__ void matmul_backward_input_kernel(const float* d_out, const float* W, float* d_input,
                                             int M, int N, int K);

__global__ void matmul_backward_weight_kernel(const float* d_out, const float* input, float* d_W,
                                              int M, int N, int K);
