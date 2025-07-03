#pragma once

enum ActivationType {
    ACT_RELU = 0,
    ACT_SIGMOID = 1,
    ACT_TANH = 2
};

__global__ void activation_relu(const float* input, const float* bias, float* output, int rows, int cols);
__global__ void activation_sigmoid(const float* input, const float* bias, float* output, int rows, int cols);
__global__ void activation_tanh(const float* input, const float* bias, float* output, int rows, int cols);
