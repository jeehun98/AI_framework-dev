// backends/cuda/ops/_common/shim/math_compat.hpp
#pragma once
#include <cmath>
#include "ai_defs.hpp"

namespace ai::cuda::shim {

// ------------------------------------------------------------
// math_compat: device / host 공용 경량 수학 함수 래퍼
// ------------------------------------------------------------

AI_RD inline float expf_compat(float x) {
#if defined(__CUDA_ARCH__)
  // device fast-math intrinsic
  return __expf(x);
#else
  return std::expf(x);
#endif
}

AI_RD inline float tanhf_compat(float x) {
#if defined(__CUDA_ARCH__)
  // device intrinsic (tanhf는 __tanhf 대신 일반 tanhf 사용)
  return tanhf(x);
#else
  return std::tanh(x);
#endif
}

AI_RD inline float sigmoid_compat(float x) {
  return 1.0f / (1.0f + expf_compat(-x));
}

AI_RD inline float relu_compat(float x) {
  return x > 0.f ? x : 0.f;
}


} // namespace ai::cuda::shim
