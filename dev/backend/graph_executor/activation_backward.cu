// activation_backward.cu
#include <cuda_runtime.h>
#include <math.h>
#include "logging_config.h"          // KPRINTF 매크로
#include "activation_backward.cuh"

__global__ void activation_backward(const float* __restrict__ grad_out,
                                    const float* __restrict__ out,
                                    float* __restrict__ grad_in,
                                    int rows, int cols, int act_type)
{
    int idx   = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    float go = grad_out[idx];
    float o  = out[idx];
    float gi = 0.0f;

    // NaN/Inf 방지 (입력 이상 시 0으로 차단)
    if (!isfinite(go) || !isfinite(o)) {
        if (idx < 1) {
            KPRINTF("[activation_backward][NaN/Inf] idx=%d | go=%.6f, o=%.6f, act=%d\n",
                    idx, go, o, act_type);
        }
        grad_in[idx] = 0.0f;
        return;
    }

    // 도함수 적용
    switch (act_type) {
        case 2: { // ReLU
            gi = (o > 0.0f) ? go : 0.0f;
            break;
        }
        case 3: { // Sigmoid (out==σ(z) ∈ [0,1])
            if (o < 0.0f || o > 1.0f) {
                if (idx < 1) {
                    KPRINTF("[sigmoid_bw][o out of range] idx=%d | o=%.6f (exp [0,1])\n", idx, o);
                }
                gi = 0.0f;
            } else {
                gi = go * o * (1.0f - o);
            }
            break;
        }
        case 4: { // Tanh (out==tanh(z) ∈ [-1,1])
            if (fabsf(o) > 1.0f) {
                if (idx < 1) {
                    KPRINTF("[tanh_bw][o out of range] idx=%d | o=%.6f (exp [-1,1])\n", idx, o);
                }
                gi = 0.0f;
            } else {
                gi = go * (1.0f - o * o);
            }
            break;
        }
        default: {
            gi = 0.0f;
            if (idx == 0) {
                KPRINTF("[activation_backward][UNKNOWN act] act_type=%d\n", act_type);
            }
            break;
        }
    }

    // 출력 안정화 (과도/비정상 값 차단)
    if (!isfinite(gi) || fabsf(gi) > 1e10f) {
        if (idx < 1) {
            KPRINTF("[activation_backward][gi abnormal] idx=%d | gi=%.6f (go=%.6f, o=%.6f, act=%d)\n",
                    idx, gi, go, o, act_type);
        }
        gi = 0.0f;
    }

    grad_in[idx] = gi;

    // 최초 한 원소만 요약 출력 (디버그 ON일 때만)
    if (idx == 0) {
        KPRINTF("[activation_backward] rows=%d cols=%d act=%d | go=%.6f o=%.6f gi=%.6f\n",
                rows, cols, act_type, go, o, gi);
    }
}
