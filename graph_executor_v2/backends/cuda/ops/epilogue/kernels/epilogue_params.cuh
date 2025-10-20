#pragma once
#include <cuda_fp16.h>
#include <cstdint>

struct EpParamsF32 {
  int M, N;
  int ld_x, ld_y;
  const float* x;
  float*       y;
  const float* bias;
  const float* resid;
  float alpha, beta;
  uint8_t use_dropout{0};
  float   p_drop{0.f};
  float   keep_scale{1.f};
  unsigned long long seed{0};
  unsigned long long offset{0};
};

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
