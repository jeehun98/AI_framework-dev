#pragma once

__global__ void conv2d_forward_kernel(const float* input, const float* kernel,
                                      float* output, int B, int H, int W,
                                      int KH, int KW, int OH, int OW);

__global__ void conv2d_backward_input_kernel(const float* d_out, const float* kernel,
                                             float* d_input, int B, int H, int W,
                                             int KH, int KW, int OH, int OW);

__global__ void conv2d_backward_kernel_kernel(const float* input, const float* d_out,
                                              float* d_kernel, int B, int H, int W,
                                              int KH, int KW, int OH, int OW);
