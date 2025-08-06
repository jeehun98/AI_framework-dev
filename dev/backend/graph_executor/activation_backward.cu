#include <cstdio>
#include <math.h>
#include "activation_backward.cuh"

__global__ void activation_backward(const float* grad_out, const float* out, float* grad_in,
                                    int rows, int cols, int act_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;

    if (idx >= total) return;

    float go = grad_out[idx];  // dL/dy
    float o  = out[idx];       // y = activation(x)
    float gi = 0.0f;           // dL/dx 초기값

    if (!isfinite(go) || !isfinite(o)) {
        if (idx == 0) {
            printf("[activation_backward][NaN/Inf] grad_out=%.6f, out=%.6f, act_type=%d\n", go, o, act_type);
        }
        grad_in[idx] = 0.0f;
        return;
    }

    switch (act_type) {
        case 2:  // ReLU
            gi = (o > 0.0f) ? go : 0.0f;
            break;

        case 3:  // Sigmoid: dy/dx = y * (1 - y)
            gi = go * o * (1.0f - o);
            break;

        case 4:  // Tanh: dy/dx = 1 - y^2
            gi = go * (1.0f - o * o);
            break;

        default:
            gi = 0.0f;
            if (idx == 0) {
                printf("[activation_backward][UNKNOWN] act_type=%d\n", act_type);
            }
            break;
    }

    grad_in[idx] = gi;
}
