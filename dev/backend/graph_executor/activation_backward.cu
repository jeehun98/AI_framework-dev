#include <cstdio>
#include <math.h>
#include "activation_backward.cuh"

__global__ void activation_backward(const float* grad_out, const float* out, float* grad_in,
                                    int rows, int cols, int act_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        float go = grad_out[idx];   // dL/dy
        float o = out[idx];         // y = activation(x)
        float gi;

        if (act_type == 3) {
            gi = go * o * (1.0f - o);  // dL/dx = dL/dy * sigmoid'(x)
        } else if (act_type == 2) {
            gi = o > 0 ? go : 0.0f;
        } else if (act_type == 4) {
            gi = go * (1.0f - o * o);
        } else {
            gi = 0.0f;
        }

        grad_in[idx] = gi;

        // ✅ 디버그 출력
        if (idx < 2) {
            printf("[activation_backward] grad_out[%d]=%.6f, out=%.6f, grad_in=%.6f\n", idx, go, o, gi);
        }
    }
}
