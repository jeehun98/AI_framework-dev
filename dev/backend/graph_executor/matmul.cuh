#pragma once
__device__ void matmul_1xN(float* input, float* weight, float* output, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += input[i] * weight[i * out_dim + j];
        }
        output[j] = sum;
    }
}
