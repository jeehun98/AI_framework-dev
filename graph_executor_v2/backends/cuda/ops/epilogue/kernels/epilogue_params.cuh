#pragma once
#include <cstdint>
#include <cuda_fp16.h>

struct EpParamsF32 {
  int M,N, ld_x, ld_y;
  const float* x;
  float* y;
  const float* bias;   // PerN or nullptr
  float alpha, beta;   // y = alpha*x + beta*y
  uint8_t act;         // 0=None,1=ReLU
  uint8_t has_bias;    // 0/1
  uint8_t use_dropout; // 0/1
  float p_drop, keep_scale;
  unsigned long long seed, offset;
};

struct EpParamsF16 {
  int M,N, ld_x, ld_y;
  const half* x;
  half* y;
  const half* bias;
  float alpha, beta;
  uint8_t act, has_bias, use_dropout;
  float p_drop, keep_scale;
  unsigned long long seed, offset;
};
