#include <cstdio>
#include <math.h>
#include "activation_backward.cuh"

__global__ void activation_backward(const float* grad_out, const float* out, float* grad_in,
                                    int rows, int cols, int act_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        float go = grad_out[idx];  // dL/dy
        float o = out[idx];        // y = activation(x)
        float gi = 0.0f;           // dL/dx 초기값

        // NaN 또는 Inf 방어
        if (!isfinite(go) || !isfinite(o)) {
            if (idx < 2) {
                printf("[activation_backward][NaN/Inf] grad_out[%d]=%.6f, out=%.6f, act_type=%d\n",
                       idx, go, o, act_type);
            }
            gi = 0.0f;  // 방어적 초기화
        } else {
            switch (act_type) {
                case 2:  // ReLU
                    gi = o > 0.0f ? go : 0.0f;
                    break;

                case 3:  // Sigmoid: dy/dx = y * (1 - y)
                    gi = go * o * (1.0f - o);
                    break;

                case 4:  // Tanh: dy/dx = 1 - y^2
                    gi = go * (1.0f - o * o);
                    break;

                case 5:  // GELU (approximate) – 추후 정확한 x 입력 시 개선 가능
                    gi = go * 1.0f;
                    break;

                case 6:  // Softplus (dy/dx = sigmoid(x), 여기선 o = sigmoid(x)라고 가정)
                    gi = go * o;
                    break;

                case 7:  // Leaky ReLU
                    gi = o > 0.0f ? go : 0.01f * go;
                    break;

                default:
                    gi = 0.0f;
                    if (idx < 2) {
                        printf("[activation_backward] ⚠️ Unknown act_type: %d\n", act_type);
                    }
                    break;
            }
        }

        grad_in[idx] = gi;

        // 디버그 출력
        if (idx < 2) {
            printf("[activation_backward] grad_out[%d]=%.6f, out=%.6f, grad_in=%.6f, act_type=%d\n",
                   idx, go, o, gi, act_type);
        }
    }
}
