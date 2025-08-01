#include <cstdio>
#include <math.h>
#include "activation_backward.cuh"

__global__ void activation_backward(const float* grad_out, const float* out, float* grad_in,
                                    int rows, int cols, int act_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        float go = grad_out[idx];  // dL/dy
        float o = out[idx];        // y = activation(x) or just f(x)
        float gi = 0.0f;           // dL/dx 초기값

        switch (act_type) {
            case 2:  // ReLU
                gi = o > 0.0f ? go : 0.0f;
                break;

            case 3:  // Sigmoid: dy/dx = y * (1 - y)
                if (!isnan(o))
                    gi = go * o * (1.0f - o);
                break;

            case 4:  // Tanh: dy/dx = 1 - y^2
                if (!isnan(o))
                    gi = go * (1.0f - o * o);
                break;

            case 5:  // GELU (approximate)
                // dy/dx ≈ 0.5 * (1 + tanh(√(2/π)(x + 0.044715x^3)))
                // 여기선 dy/dy 기준이므로 skip하거나 정확한 x 전달 필요
                gi = go * 1.0f;  // 임시 처리, 실제 x 값이 필요함
                break;

            case 6:  // Softplus: dy/dx = sigmoid(x)
                gi = go * o;  // o = sigmoid(x)로 가정
                break;

            case 7:  // Leaky ReLU
                gi = o > 0.0f ? go : 0.01f * go;
                break;

            default:
                // 알 수 없는 타입
                gi = 0.0f;
                if (idx < 2) {
                    printf("[activation_backward] ⚠️ Unknown act_type: %d\n", act_type);
                }
                break;
        }

        grad_in[idx] = gi;

        // ✅ 디버그 출력
        if (idx < 2) {
            printf("[activation_backward] grad_out[%d]=%.6f, out=%.6f, grad_in=%.6f, act_type=%d\n",
                   idx, go, o, gi, act_type);
        }
    }
}
