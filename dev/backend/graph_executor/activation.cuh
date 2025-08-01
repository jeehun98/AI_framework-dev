#pragma once

enum ActivationType {
    ACT_NONE = 0,
    ACT_RELU = 2,
    ACT_SIGMOID = 3,
    ACT_TANH = 4
};


__global__ void activation_relu(const float* input, const float* bias, float* output, int rows, int cols);
__global__ void activation_sigmoid(const float* input, const float* bias, float* output, int rows, int cols);
__global__ void activation_tanh(const float* input, const float* bias, float* output, int rows, int cols);
