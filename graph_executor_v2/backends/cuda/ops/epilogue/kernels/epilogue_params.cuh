// kernels/epilogue_params.cuh

#pragma once
#include <cstdint>

struct EpParams {
  // dims / ld
  int M, N;
  int ld_x, ld_y;
  // ptrs
  const float* x;
  float*       y;
  const float* bias; // PerN

  // scalars
  float alpha, beta;
  uint8_t act;   // 0:None, 1:ReLU
  uint8_t has_bias; // 0/1
};


/*
#pragma once
#include <cstdint>

struct EpParams {
  // dims / ld
  int M, N;
  int ld_x, ld_y;
  // ptrs
  const void* x;  void* y;
  const void* bias;  const void* resid;  void* z;
  const uint8_t* mask;
  // strides (런처가 계산해 채움)
  long long sx_m, sx_n, sy_m, sy_n, sb_m, sb_n, sr_m, sr_n, sz_m, sz_n;
  // scalars
  float alpha, beta, act_alpha, p_drop, keep_scale, clamp_min, clamp_max;
  // rng
  unsigned long long seed, offset;
  // flags/enums (8-bit들)
  uint32_t opmask;      // BIAS|SAVEZ|ACT|DROP|RESID|BETA|CLAMP|QUANT|MASK
  uint8_t act, resid_k, quant, bias_k, x_type, y_type, b_type;
};

*/