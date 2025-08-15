#pragma once
#include <cuda_runtime.h>

// ✅ 내부 act_type 식별자
enum {
  ACT_IDENTITY = 0,
  ACT_RELU     = 1,
  ACT_SIGMOID  = 2,
  ACT_TANH     = 3,
  ACT_LEAKY    = 4,
  ACT_ELU      = 5,
  ACT_GELU     = 6,
  ACT_SILU     = 7
};

// 블록 구성 (x축이 warp 방향 → coalesced)
#ifndef ACT_BLOCK_X
#define ACT_BLOCK_X 32
#endif
#ifndef ACT_BLOCK_Y
#define ACT_BLOCK_Y 8
#endif

// 확장된 런처 시그니처(추천)
void launch_activation_forward(const float* in, const float* bias, float* out,
                               int rows, int cols, int act_type,
                               float alpha, int gelu_tanh_flag,
                               cudaStream_t stream);

void launch_activation_backward(const float* grad_out,
                                const float* in,      // pre-activation z
                                const float* out,     // f(z)
                                float* grad_in,
                                int rows, int cols, int act_type,
                                float alpha, int gelu_tanh_flag,
                                cudaStream_t stream);