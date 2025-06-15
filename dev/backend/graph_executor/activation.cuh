#pragma once
#include <math.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float tanh_custom(float x) {
    return tanhf(x);
}

__device__ void activation_vec(float* input, float* output, int dim, int type) {
    for (int i = 0; i < dim; ++i) {
        if (type == 2)        output[i] = sigmoid(input[i]);
        else if (type == 3)   output[i] = relu(input[i]);
        else if (type == 4)   output[i] = tanh_custom(input[i]);
    }
}
