#pragma once
#include "regemm/api.h"

namespace regemm {


// ------------------------------------------------------------
// 컴파일타임 템플릿 버전 (branchless)
// ------------------------------------------------------------
template<ActKind AK>
__device__ __forceinline__ float act_apply(float x) {
  return x; // 기본값 (None)
}

template<>
__device__ __forceinline__ float act_apply<ActKind::None>(float x) {
  return x;
}

template<>
__device__ __forceinline__ float act_apply<ActKind::ReLU>(float x) {
  return x > 0.f ? x : 0.f;
}

template<>
__device__ __forceinline__ float act_apply<ActKind::LeakyReLU>(float x) {
  const float a = 0.01f; // slope
  return x > 0.f ? x : a * x;
}

template<>
__device__ __forceinline__ float act_apply<ActKind::GELU>(float x) {
  // tanh 근사 버전
  const float k0 = 0.7978845608f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  float x3 = x * x * x;
  float t  = k0 * (x + k1 * x3);
  return 0.5f * x * (1.f + tanhf(t));
}

template<>
__device__ __forceinline__ float act_apply<ActKind::Sigmoid>(float x) {
  return 1.f / (1.f + expf(-x));
}

template<>
__device__ __forceinline__ float act_apply<ActKind::Tanh>(float x) {
  return tanhf(x);
}

// ------------------------------------------------------------
// 런타임 버전 (smoke 커널에서 사용)
// ------------------------------------------------------------
static __device__ __forceinline__ float apply_act_runtime(float x, ActKind k) {
  switch (k) {
    case ActKind::ReLU:     return act_apply<ActKind::ReLU>(x);
    case ActKind::LeakyReLU:return act_apply<ActKind::LeakyReLU>(x);
    case ActKind::GELU:     return act_apply<ActKind::GELU>(x);
    case ActKind::Sigmoid:  return act_apply<ActKind::Sigmoid>(x);
    case ActKind::Tanh:     return act_apply<ActKind::Tanh>(x);
    case ActKind::None:
    default:                return act_apply<ActKind::None>(x);
  }
}

} // namespace regemm