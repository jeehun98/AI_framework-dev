#include <cstdio>
#include <math.h>
#include "activation_backward.cuh"

__global__ void activation_backward(const float* grad_out, const float* out, float* grad_in,
                                    int rows, int cols, int act_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    float go = grad_out[idx];
    float o  = out[idx];
    float gi = 0.0f;

    // 🔍 NaN 또는 Inf 방지
    if (!isfinite(go) || !isfinite(o)) {
        if (idx < 10) {
            printf("[activation_backward][NaN/Inf] idx=%d | grad_out=%.6f, out=%.6f, act_type=%d\n", idx, go, o, act_type);
        }
        grad_in[idx] = 0.0f;
        return;
    }

    // 🔍 o 값 범위 검사 및 도함수 계산
    switch (act_type) {
        case 2:  // ReLU
            gi = (o > 0.0f) ? go : 0.0f;
            break;

        case 3:  // Sigmoid
            if (o < 0.0f || o > 1.0f) {
                if (idx < 10) {
                    printf("[sigmoid_backward][o out of range] idx=%d | o=%.6f (expected [0,1])\n", idx, o);
                }
                gi = 0.0f;
            } else {
                gi = go * o * (1.0f - o);
            }
            break;

        case 4:  // Tanh
            if (fabsf(o) > 1.0f) {
                if (idx < 10) {
                    printf("[tanh_backward][o out of range] idx=%d | o=%.6f (expected [-1,1])\n", idx, o);
                }
                gi = 0.0f;
            } else {
                gi = go * (1.0f - o * o);
            }
            break;

        default:
            gi = 0.0f;
            if (idx == 0)
                printf("[activation_backward][UNKNOWN] act_type=%d\n", act_type);
            break;
    }

    // 🔍 grad_input 값 검사
    if (!isfinite(gi) || fabsf(gi) > 1e10f) {
        if (idx < 10) {
            printf("[activation_backward][gi abnormal] idx=%d | gi=%.6f (from go=%.6f, o=%.6f, act_type=%d)\n", idx, gi, go, o, act_type);
        }
        gi = 0.0f;
    }

    grad_in[idx] = gi;
}
