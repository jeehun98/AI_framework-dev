// backends/cuda/ops/_common/shim/activations.hpp
#pragma once
#include <cmath>
#include "ai_defs.hpp"     // AI_RD (__host__/__device__)
#include "enums.hpp"       // ActKind
#include "math_compat.hpp" // expf_compat, tanhf_compat

namespace ai::cuda::shim {

// ===== (1) 템플릿: Forward (float 전용) =====
template<ActKind AK>
AI_RD float act_apply(float x) { return x; }

template<> AI_RD inline float act_apply<ActKind::None>(float x)      { return x; }
template<> AI_RD inline float act_apply<ActKind::ReLU>(float x)      { return x > 0.f ? x : 0.f; }
template<> AI_RD inline float act_apply<ActKind::LeakyReLU>(float x) { const float a = 0.01f; return x > 0.f ? x : a * x; }
template<> AI_RD inline float act_apply<ActKind::GELU>(float x) {
  // tanh approximation: 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715 x^3)))
  const float c  = 0.7978845608f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  float x2 = x * x, x3 = x2 * x;
  float t  = fmaf(c * k1, x3, c * x);
  return 0.5f * x * (1.f + tanhf_compat(t));
}
template<> AI_RD inline float act_apply<ActKind::Sigmoid>(float x) {
  // branchless overflow-safe sigmoid
  if (x >= 0.f) { float z = expf_compat(-x); return 1.f / (1.f + z); }
  else          { float z = expf_compat(x);  return z / (1.f + z);  }
}
template<> AI_RD inline float act_apply<ActKind::Tanh>(float x)     { return tanhf_compat(x); }

// ===== (2) 템플릿: Forward (leaky 파라미터 지원) =====
template<ActKind AK>
AI_RD float act_apply(float x, float leaky) { return act_apply<AK>(x); }

template<>
AI_RD inline float act_apply<ActKind::LeakyReLU>(float x, float leaky) {
  return x > 0.f ? x : leaky * x;
}

// ===== (3) 템플릿: Derivative =====
template<ActKind AK>
AI_RD float act_deriv(float /*x*/, float /*leaky*/) { return 1.f; }

template<> AI_RD inline float act_deriv<ActKind::None>(float, float)            { return 1.f; }
template<> AI_RD inline float act_deriv<ActKind::ReLU>(float x, float)          { return x > 0.f ? 1.f : 0.f; }
template<> AI_RD inline float act_deriv<ActKind::LeakyReLU>(float x, float a)   { return x > 0.f ? 1.f : a; }
template<> AI_RD inline float act_deriv<ActKind::GELU>(float x, float) {
  const float c  = 0.7978845608f;
  const float k1 = 0.044715f;
  float x2  = x * x, x3 = x2 * x;
  float t   = fmaf(c * k1, x3, c * x);
  float th  = tanhf_compat(t);
  // clamp to avoid NaN from tiny negative sech^2 under fp rounding
  if (th >  0.999999f) th =  0.999999f;
  if (th < -0.999999f) th = -0.999999f;
  float sech2 = 1.f - th * th;
  float dt    = c * (1.f + 3.f * k1 * x2);
  return 0.5f * (1.f + th) + 0.5f * x * sech2 * dt;
}
template<> AI_RD inline float act_deriv<ActKind::Sigmoid>(float x, float) {
  float s;
  if (x >= 0.f) { float z = expf_compat(-x); s = 1.f / (1.f + z); }
  else          { float z = expf_compat(x);  s = z   / (1.f + z); }
  return s * (1.f - s);
}
template<> AI_RD inline float act_deriv<ActKind::Tanh>(float x, float) {
  float th = tanhf_compat(x);
  return 1.f - th * th;
}

// ===== (4) 공통 ReLU류 =====
AI_RD inline float relu_like(float x, float leaky)    { return x > 0.f ? x : leaky * x; }
AI_RD inline float d_relu_like(float x, float leaky)  { return x > 0.f ? 1.f : leaky; }

// ===== (5) 런타임 Forward =====
AI_RD inline float apply_act_runtime(float x, ActKind k, float leaky = 0.0f) {
  switch (k) {
    case ActKind::ReLU:      return x > 0.f ? x : 0.f;
    case ActKind::LeakyReLU: return x > 0.f ? x : leaky * x;
    case ActKind::GELU: {
      const float c = 0.7978845608f, k1 = 0.044715f;
      float t = fmaf(c * k1, x * x * x, c * x);
      return 0.5f * x * (1.f + tanhf_compat(t));
    }
    case ActKind::Sigmoid:
      if (x >= 0.f) { float z = expf_compat(-x); return 1.f / (1.f + z); }
      else          { float z = expf_compat(x);  return z / (1.f + z);  }
    case ActKind::Tanh:      return tanhf_compat(x);
    case ActKind::None: default: return x;
  }
}

// ===== (6) 런타임 Backward: gZ = gY * act'(Z) =====
AI_RD inline float apply_act_grad_runtime(float Z, float gY, ActKind k, float leaky) {
  switch (k) {
    case ActKind::ReLU:      return gY * (Z > 0.f ? 1.f : 0.f);
    case ActKind::LeakyReLU: return gY * (Z > 0.f ? 1.f : leaky);
    case ActKind::GELU: {
      const float c = 0.7978845608f, k1 = 0.044715f;
      float x2 = Z * Z;
      float t  = fmaf(c * k1, Z * x2, c * Z);
      float th = tanhf_compat(t);
      float sech2 = 1.f - th * th;
      float dt = c * (1.f + 3.f * k1 * x2);
      return gY * (0.5f * (1.f + th) + 0.5f * Z * sech2 * dt);
    }
    case ActKind::Sigmoid: {
      float s;
      if (Z >= 0.f) { float z = expf_compat(-Z); s = 1.f / (1.f + z); }
      else          { float z = expf_compat(Z);  s = z   / (1.f + z); }
      return gY * s * (1.f - s);
    }
    case ActKind::Tanh: {
      float th = tanhf_compat(Z);
      return gY * (1.f - th * th);
    }
    case ActKind::None: default: return gY;
  }
}

} // namespace ai::cuda::shim
