
#pragma once

__global__ void matmul_backward_input(const float* d_out, const float* W_T, float* d_input, int M, int N, int K);
__global__ void matmul_backward_weight(const float* input_T, const float* d_out, float* d_weight, int M, int N, int K);
__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols);
