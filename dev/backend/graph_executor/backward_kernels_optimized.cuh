#pragma once

#define TILE_WIDTH 16

__global__ void matmul_backward_input_shared(const float* __restrict__ d_out,
                                             const float* __restrict__ W_T,
                                             float* __restrict__ d_input,
                                             int M, int N, int K);

__global__ void matmul_backward_weight_shared(const float* __restrict__ input,
                                              const float* __restrict__ d_out,
                                              float* __restrict__ d_weight,
                                              int M, int N, int K);

__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols);
__global__ void add_backward_input(const float* d_out, float* d_input, int size);
__global__ void fill_gradient(float* grad, int total_size, float value);

__global__ void matmul_backward_input_simple(const float* __restrict__ d_out, 
                                             const float* __restrict__ W_T, 
                                             float* __restrict__ d_input, 
                                             int M, int N, int K);
