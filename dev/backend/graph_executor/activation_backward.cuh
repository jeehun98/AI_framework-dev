
#pragma once

__global__ void activation_backward(const float* d_out, const float* output, float* d_input, int rows, int cols, int act_type);
