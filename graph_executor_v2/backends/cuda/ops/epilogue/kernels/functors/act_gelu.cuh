#pragma once
#include <cuda_fp16.h>

// fast-approx GELU (tanh) — widely used
__device__ __forceinline__ float gelu_f(float x) {
  const float kAlpha = 0.7978845608028654f;   // sqrt(2/pi)
  const float kBeta  = 0.044715f;
  float x3 = x * x * x;
  float t  = kAlpha * (x + kBeta * x3);
  float y  = 0.5f * x * (1.f + tanhf(t));
  return y;
}
__device__ __forceinline__ half gelu_h(half xh) {
  float x = __half2float(xh);
  float y = gelu_f(x);
  return __float2half(y);
}
