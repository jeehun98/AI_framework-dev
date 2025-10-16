// backends/cuda/ops/gemm/detail/activations.h
#pragma once
#include "api.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>  // host 테스트용 (fmaf, tanhf 등)

// ------------------------------------------------------------
// 호스트 단위테스트를 돌리고 싶으면 컴파일 옵션에
//   -DREGEMM_TEST_ON_HOST
// 를 추가하세요.
// ------------------------------------------------------------
#ifdef REGEMM_TEST_ON_HOST
  #define RD __host__ __device__
#else
  #define RD __device__
#endif

namespace regemm {

// ============================================================
// (0) 디바이스/호스트 호환 보조 함수
// ============================================================
RD __forceinline__ float expf_compat(float x) {
#if defined(__CUDA_ARCH__)
  return __expf(x);
#else
  return std::expf(x);
#endif
}

RD __forceinline__ float tanhf_compat(float x) {
#if defined(__CUDA_ARCH__)
  return tanhf(x);
#else
  return std::tanh(x);
#endif
}

// ============================================================
// (1) 컴파일타임 템플릿: Forward 기본형 (float 전용)
// ============================================================
template<ActKind AK>
RD __forceinline__ float act_apply(float x) {
  // None default
  return x;
}

template<>
RD __forceinline__ float act_apply<ActKind::None>(float x) { return x; }

template<>
RD __forceinline__ float act_apply<ActKind::ReLU>(float x) {
  return x > 0.f ? x : 0.f;
}

template<>
RD __forceinline__ float act_apply<ActKind::LeakyReLU>(float x) {
  // legacy slope = 0.01f
  const float a = 0.01f;
  return x > 0.f ? x : a * x;
}

template<>
RD __forceinline__ float act_apply<ActKind::GELU>(float x) {
  // tanh approximation: 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
  const float c  = 0.7978845608f;  // sqrt(2/pi)
  const float k1 = 0.044715f;
  float x2 = x * x;
  float x3 = x2 * x;
  float t  = fmaf(c * k1, x3, c * x);
  return 0.5f * x * (1.f + tanhf_compat(t));
}

template<>
RD __forceinline__ float act_apply<ActKind::Sigmoid>(float x) {
  // 수치 안정형 시그모이드
  if (x >= 0.f) {
    float z = expf_compat(-x);
    return 1.f / (1.f + z);
  } else {
    float z = expf_compat(x);
    return z / (1.f + z);
  }
}

template<>
RD __forceinline__ float act_apply<ActKind::Tanh>(float x) {
  return tanhf_compat(x);
}

// ============================================================
// (2) 컴파일타임 템플릿: Forward (leaky 파라미터 지원 오버로드)
// ============================================================
template<ActKind AK>
RD __forceinline__ float act_apply(float x, float /*leaky*/) {
  // 기본은 leaky 미사용
  return act_apply<AK>(x);
}

template<>
RD __forceinline__ float act_apply<ActKind::LeakyReLU>(float x, float leaky) {
  return x > 0.f ? x : leaky * x;
}

// ============================================================
// (3) 컴파일타임 템플릿: Derivative (Backward용) — d act(x) / dx
// ============================================================
template<ActKind AK>
RD __forceinline__ float act_deriv(float /*x*/, float /*leaky*/) {
  // None default
  return 1.f;
}

template<>
RD __forceinline__ float act_deriv<ActKind::None>(float /*x*/, float /*leaky*/) {
  return 1.f;
}

template<>
RD __forceinline__ float act_deriv<ActKind::ReLU>(float x, float /*leaky*/) {
  return x > 0.f ? 1.f : 0.f;
}

template<>
RD __forceinline__ float act_deriv<ActKind::LeakyReLU>(float x, float leaky) {
  return x > 0.f ? 1.f : leaky;
}

template<>
RD __forceinline__ float act_deriv<ActKind::GELU>(float x, float /*leaky*/) {
  // d/dx [ 0.5*x*(1 + tanh(t)) ] with t = c*(x + k1*x^3)
  const float c  = 0.7978845608f;  // sqrt(2/pi)
  const float k1 = 0.044715f;

  float x2  = x * x;
  float x3  = x2 * x;
  float t   = fmaf(c * k1, x3, c * x);      // c*(x + k1*x^3)
  float th  = tanhf_compat(t);
  // clamp to avoid tiny negative due to FP error when |t| is large
  if (th >  0.999999f) th =  0.999999f;
  if (th < -0.999999f) th = -0.999999f;

  float sech2 = 1.f - th * th;              // sech^2(t) = 1 - tanh^2(t)
  float dt    = c * (1.f + 3.f * k1 * x2);  // dt/dx
  // dy/dx = 0.5*(1 + tanh(t)) + 0.5*x*sech^2(t)*dt
  return 0.5f * (1.f + th) + 0.5f * x * sech2 * dt;
}

template<>
RD __forceinline__ float act_deriv<ActKind::Sigmoid>(float x, float /*leaky*/) {
  // s(x) * (1 - s(x)) (안정형 s 사용)
  float s;
  if (x >= 0.f) {
    float z = expf_compat(-x);
    s = 1.f / (1.f + z);
  } else {
    float z = expf_compat(x);
    s = z / (1.f + z);
  }
  return s * (1.f - s);
}

template<>
RD __forceinline__ float act_deriv<ActKind::Tanh>(float x, float /*leaky*/) {
  float th = tanhf_compat(x);
  return 1.f - th * th;
}

// ============================================================
// (4) 공통화: ReLU/LeakyReLU (forward & deriv)
// ============================================================
RD __forceinline__ float relu_like(float x, float leaky) {
  // leaky == 0 -> ReLU, 0<leaky<1 -> LeakyReLU
  return x > 0.f ? x : leaky * x;
}

RD __forceinline__ float d_relu_like(float x, float leaky) {
  return x > 0.f ? 1.f : leaky;
}

// ============================================================
// (5) 런타임 버전: Forward
// ============================================================
// ★ 핵심: 3-인자 버전을 먼저 정의하고, 2-인자 래퍼는 아래에서 호출하도록 하여
//    "too many arguments"나 오버로드 가려짐 문제를 방지합니다.
static RD __forceinline__ float apply_act_runtime(float x, ActKind k, float leaky) {
  switch (k) {
    case ActKind::ReLU:      return (x > 0.f ? x : 0.f);
    case ActKind::LeakyReLU: return (x > 0.f ? x : leaky * x);
    case ActKind::GELU: {
      const float c  = 0.7978845608f;  // sqrt(2/pi)
      const float k1 = 0.044715f;
      float t = fmaf(c * k1, x * x * x, c * x);
      return 0.5f * x * (1.f + tanhf_compat(t));
    }
    case ActKind::Sigmoid: {
      if (x >= 0.f) {
        float z = expf_compat(-x);
        return 1.f / (1.f + z);
      } else {
        float z = expf_compat(x);
        return z / (1.f + z);
      }
    }
    case ActKind::Tanh:      return tanhf_compat(x);
    case ActKind::None:
    default:                 return x;
  }
}

// 2-인자 래퍼: leaky 미지정 시 0으로 취급 (과거 호출부 호환)
static RD __forceinline__ float apply_act_runtime(float x, ActKind k) {
  return apply_act_runtime(x, k, 0.0f);
}

// ============================================================
// (6) 런타임 버전: Backward (gZ = gY * act'(Z))
// ============================================================
static RD __forceinline__ float apply_act_grad_runtime(float Z, float gY, ActKind k, float leaky) {
  switch (k) {
    case ActKind::ReLU:      return gY * (Z > 0.f ? 1.f : 0.f);
    case ActKind::LeakyReLU: return gY * (Z > 0.f ? 1.f : leaky);
    case ActKind::GELU: {
      const float c  = 0.7978845608f;  // sqrt(2/pi)
      const float k1 = 0.044715f;
      float x2 = Z * Z;
      float t  = fmaf(c * k1, Z * x2, c * Z);
      float th = tanhf_compat(t);
      float sech2 = 1.f - th * th;
      float dt = c * (1.f + 3.f * k1 * x2);
      return gY * (0.5f * (1.f + th) + 0.5f * Z * sech2 * dt);
    }
    case ActKind::Sigmoid: {
      // s*(1-s) (안정형)
      float s;
      if (Z >= 0.f) {
        float z = expf_compat(-Z);
        s = 1.f / (1.f + z);
      } else {
        float z = expf_compat(Z);
        s = z / (1.f + z);
      }
      return gY * s * (1.f - s);
    }
    case ActKind::Tanh: {
      float th = tanhf_compat(Z);
      return gY * (1.f - th * th);
    }
    case ActKind::None:
    default:                 return gY;
  }
}

} // namespace regemm

#ifdef REGEMM_TEST_ON_HOST
#undef RD
#endif
