#pragma once
#include <cuda_runtime.h>

// run_graph의 OpType을 그대로 쓰는 경우 (RELU=2, SIGMOID=3, TANH=4)
enum ActivationType {
    ACT_RELU    = 2,
    ACT_SIGMOID = 3,
    ACT_TANH    = 4
};

// 블록 구성 (x축이 warp 방향 → coalesced)
#ifndef ACT_BLOCK_X
#define ACT_BLOCK_X 32
#endif
#ifndef ACT_BLOCK_Y
#define ACT_BLOCK_Y 8
#endif

// Forward: out = act(input + (bias ? bias[col] : 0))
void launch_activation_forward(const float* in, const float* bias, float* out,
                               int rows, int cols, int act_type,
                               cudaStream_t stream = 0);

// Backward: grad_in = grad_out * act'(out)
//           (주의: out은 forward 결과값)
void launch_activation_backward(const float* grad_out, const float* out, float* grad_in,
                                int rows, int cols, int act_type,
                                cudaStream_t stream = 0);
