// backends/cuda/ops/_common/shim/bias.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "enums.hpp"  // BiasKind
#include "ai_defs.hpp"

// 단위테스트/디버그 플래그 표준화
#if defined(REGEMM_TEST_ON_HOST) && !defined(AI_SHIM_TEST_ON_HOST)
#define AI_SHIM_TEST_ON_HOST
#endif
#if defined(REGEMM_DEBUG) && !defined(AI_SHIM_DEBUG)
#define AI_SHIM_DEBUG
#endif

#ifdef AI_SHIM_TEST_ON_HOST
  #define AI_RD __host__ __device__ __forceinline__
#else
  #define AI_RD __device__ __forceinline__
#endif

namespace ai::cuda::shim {

// ---------- 메타: bias 크기 계산 ----------
AI_RD inline std::size_t expected_bias_elems(int M, int N, BiasKind k) {
  switch (k) {
    case BiasKind::None:   return 0;
    case BiasKind::Scalar: return 1;
    case BiasKind::PerM:   return (M > 0) ? static_cast<std::size_t>(M) : 0;
    case BiasKind::PerN:   return (N > 0) ? static_cast<std::size_t>(N) : 0;
    default:               return 0;
  }
}

// ---------- 디바이스/호스트 호환 단일 원소 로드 ----------
AI_RD inline float ld1_compat(const float* p) {
#if defined(__CUDA_ARCH__)
  // 읽기 전용 캐시 힌트. (새 아키텍처에선 일반 로드와 큰 차이 없지만 호환 유지)
  return __ldg(p);
#else
  return *p;
#endif
}

// ---------- 로우-레벨 포인터 기반 로드 ----------
AI_RD inline float load_bias_ptr(const void* bias, BiasKind kind,
                                 int m, int n, int M, int N) {
  if (!bias || kind == BiasKind::None) return 0.f;

#ifdef AI_SHIM_DEBUG
  if (kind == BiasKind::PerM) {
    // m in [0, M)
    if (!(m >= 0 && M > 0 && m < M)) asm volatile(""); // 방해 최소, 어설트 대체
  } else if (kind == BiasKind::PerN) {
    // n in [0, N)
    if (!(n >= 0 && N > 0 && n < N)) asm volatile("");
  }
#endif

  const float* b = reinterpret_cast<const float*>(bias);
  switch (kind) {
    case BiasKind::Scalar: return ld1_compat(b);
    case BiasKind::PerM:   return ld1_compat(b + m);
    case BiasKind::PerN:   return ld1_compat(b + n);
    case BiasKind::None:
    default:               return 0.f;
  }
}

// ---------- Params 개체(POD)에 의존하는 제네릭 로드 ----------
// P는 다음 필드를 가져야 한다: bias(void*), bias_kind(BiasKind), M, N
template <typename P>
AI_RD inline float load_bias(const P& p, int m, int n) {
  return load_bias_ptr(p.bias, p.bias_kind, m, n, p.M, p.N);
}

// ---------- 누산값에 bias 더하기 ----------
template <typename P>
AI_RD inline float add_bias(float z, const P& p, int m, int n) {
  return z + load_bias(p, m, n);
}

// 1D 텐서 길이로 BiasKind 추론
[[nodiscard]] inline BiasKind
infer_bias_kind_1d_lenMN(const ai::Tensor* Bias, int64_t M, int64_t N) noexcept {
  if (!Bias || !Bias->data)                return BiasKind::None;
  if (Bias->desc.shape.size() != 1)        return BiasKind::None;
  const int64_t len = Bias->desc.shape[0];
  if (len == 1) return BiasKind::Scalar;
  if (len == N) return BiasKind::PerN;
  if (len == M) return BiasKind::PerM;
  return BiasKind::None;
}

[[nodiscard]] inline BiasKind
deduce_bias_kind_from_forward(const ai::Tensor* bias_like,
                              int64_t M, int64_t N) noexcept {
  return infer_bias_kind_1d_lenMN(bias_like, M, N);
}

} // namespace ai::cuda::shim

#ifdef AI_SHIM_TEST_ON_HOST
#undef AI_RD
#endif
