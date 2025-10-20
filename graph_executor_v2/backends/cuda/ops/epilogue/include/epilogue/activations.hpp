#pragma once
#include "epilogue/config.hpp"
#include <cuda_fp16.h>

namespace epilogue {

// tanh 기반 fast GELU 근사
__device__ __forceinline__ float gelu_fast(float x) {
  const float k0 = 0.7978845608f;   // sqrt(2/pi)
  const float k1 = 0.044715f;
  float y = x * (1.0f + 0.5f * tanhf(k0 * (x + k1 * x * x * x)));
  return y;
}

template<ActKind AK>
__device__ __forceinline__ float act(float v, float leaky) {
  if constexpr (AK == ActKind::None)      return v;
  if constexpr (AK == ActKind::ReLU)      return v > 0.f ? v : 0.f;
  if constexpr (AK == ActKind::LeakyReLU) return v > 0.f ? v : leaky * v;
  if constexpr (AK == ActKind::GELU)      return gelu_fast(v);
  if constexpr (AK == ActKind::Sigmoid)   return 1.f / (1.f + expf(-v));
  if constexpr (AK == ActKind::Tanh)      return tanhf(v);
  return v;
}

} // namespace epilogue
