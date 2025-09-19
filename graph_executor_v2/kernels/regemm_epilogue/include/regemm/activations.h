#pragma once
#include "regemm/api.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace regemm {

// ============================================================
// (1) 컴파일타임 템플릿: Forward (기존 유지)
// ============================================================
template<ActKind AK>
__device__ __forceinline__ float act_apply(float x) {
  return x; // None default
}

template<>
__device__ __forceinline__ float act_apply<ActKind::None>(float x) { return x; }

template<>
__device__ __forceinline__ float act_apply<ActKind::ReLU>(float x) {
  return x > 0.f ? x : 0.f;
}

template<>
__device__ __forceinline__ float act_apply<ActKind::LeakyReLU>(float x) {
  const float a = 0.01f; // legacy default slope
  return x > 0.f ? x : a * x;
}

template<>
__device__ __forceinline__ float act_apply<ActKind::GELU>(float x) {
  // tanh approximation
  const float k0 = 0.7978845608f;
  const float k1 = 0.044715f;
  float x3 = x * x * x;
  float t  = k0 * (x + k1 * x3);
  return 0.5f * x * (1.f + tanhf(t));
}

template<>
__device__ __forceinline__ float act_apply<ActKind::Sigmoid>(float x) {
  return 1.f / (1.f + __expf(-x));
}

template<>
__device__ __forceinline__ float act_apply<ActKind::Tanh>(float x) {
  return tanhf(x);
}

// ============================================================
// (2) 컴파일타임 템플릿: Forward (leaky 파라미터 지원 오버로드)
// ============================================================
template<ActKind AK>
__device__ __forceinline__ float act_apply(float x, float /*leaky*/) {
  // 기본은 leaky 미사용
  return act_apply<AK>(x);
}

template<>
__device__ __forceinline__ float act_apply<ActKind::LeakyReLU>(float x, float leaky) {
  return x > 0.f ? x : leaky * x;
}

// ============================================================
// (3) 컴파일타임 템플릿: Derivative (Backward용)
//     반환: d act(x) / dx
// ============================================================
template<ActKind AK>
__device__ __forceinline__ float act_deriv(float /*x*/, float /*leaky*/) {
  // None default
  return 1.f;
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::None>(float /*x*/, float /*leaky*/) {
  return 1.f;
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::ReLU>(float x, float /*leaky*/) {
  return x > 0.f ? 1.f : 0.f;
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::LeakyReLU>(float x, float leaky) {
  return x > 0.f ? 1.f : leaky;
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::GELU>(float x, float /*leaky*/) {
  // tanh approx derivative
  const float c  = 0.7978845608f;
  const float k1 = 0.044715f;
  float x2  = x * x;
  float x3  = x2 * x;
  float t   = c * (x + k1 * x3);
  float th  = tanhf(t);
  float sech2 = 1.f - th * th;               // sech^2(t) = 1 - tanh^2(t)
  float dt  = c * (1.f + 3.f * k1 * x2);     // d/dx of t
  // dy/dx = 0.5*(1 + tanh(t)) + 0.5*x*sech^2(t)*dt
  return 0.5f * (1.f + th) + 0.5f * x * sech2 * dt;
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::Sigmoid>(float x, float /*leaky*/) {
  float s = 1.f / (1.f + __expf(-x));
  return s * (1.f - s);
}

template<>
__device__ __forceinline__ float act_deriv<ActKind::Tanh>(float x, float /*leaky*/) {
  float th = tanhf(x);
  return 1.f - th * th;
}

// ============================================================
// (4) 런타임 버전: Forward
//     (기존 runtime + leaky 파라미터 지원)
// ============================================================
static __device__ __forceinline__ float apply_act_runtime(float x, ActKind k) {
  switch (k) {
    case ActKind::ReLU:      return act_apply<ActKind::ReLU>(x);
    case ActKind::LeakyReLU: return act_apply<ActKind::LeakyReLU>(x);
    case ActKind::GELU:      return act_apply<ActKind::GELU>(x);
    case ActKind::Sigmoid:   return act_apply<ActKind::Sigmoid>(x);
    case ActKind::Tanh:      return act_apply<ActKind::Tanh>(x);
    case ActKind::None:
    default:                 return act_apply<ActKind::None>(x);
  }
}

static __device__ __forceinline__ float apply_act_runtime(float x, ActKind k, float leaky) {
  switch (k) {
    case ActKind::ReLU:      return act_apply<ActKind::ReLU>(x);
    case ActKind::LeakyReLU: return act_apply<ActKind::LeakyReLU>(x, leaky);
    case ActKind::GELU:      return act_apply<ActKind::GELU>(x);
    case ActKind::Sigmoid:   return act_apply<ActKind::Sigmoid>(x);
    case ActKind::Tanh:      return act_apply<ActKind::Tanh>(x);
    case ActKind::None:
    default:                 return act_apply<ActKind::None>(x);
  }
}

// ============================================================
// (5) 런타임 버전: Backward (gZ = gY * act'(Z))
// ============================================================
static __device__ __forceinline__ float apply_act_grad_runtime(float Z, float gY, ActKind k, float leaky) {
  switch (k) {
    case ActKind::ReLU:      return gY * act_deriv<ActKind::ReLU>(Z, leaky);
    case ActKind::LeakyReLU: return gY * act_deriv<ActKind::LeakyReLU>(Z, leaky);
    case ActKind::GELU:      return gY * act_deriv<ActKind::GELU>(Z, leaky);
    case ActKind::Sigmoid:   return gY * act_deriv<ActKind::Sigmoid>(Z, leaky);
    case ActKind::Tanh:      return gY * act_deriv<ActKind::Tanh>(Z, leaky);
    case ActKind::None:
    default:                 return gY * act_deriv<ActKind::None>(Z, leaky);
  }
}

} // namespace regemm
