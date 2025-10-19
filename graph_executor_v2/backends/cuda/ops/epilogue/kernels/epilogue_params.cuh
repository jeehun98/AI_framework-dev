#pragma once
#include <cstdint>
#include <cuda_fp16.h>

// FP32 파라미터 (POD)
struct EpParamsF32 {
  int M, N;
  int ld_x, ld_y;
  const float* x;  // [M,?]
  float*       y;  // [M,?]
  const float* bias;   // [N] or nullptr
  const float* resid;  // [M,?] or nullptr
  float alpha, beta;   // blend
  // dropout
  uint8_t use_dropout{0};
  float   p_drop{0.f};
  float   keep_scale{1.f};
  // RNG
  unsigned long long seed{0};
  unsigned long long offset{0};
};

// FP16 파라미터 (POD)
struct EpParamsF16 {
  int M, N;
  int ld_x, ld_y;
  const half* x;
  half*       y;
  const half* bias;
  const half* resid;
  float alpha, beta;
  uint8_t use_dropout{0};
  float   p_drop{0.f};
  float   keep_scale{1.f};
  unsigned long long seed{0};
  unsigned long long offset{0};
};
