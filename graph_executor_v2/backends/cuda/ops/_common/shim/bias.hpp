// backends/cuda/ops/_common/shim/bias.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "ai_defs.hpp"     // AI_RD, AI_INLINE
#include "enums.hpp"       // BiasKind
#include "ai_tensor.hpp"   // Tensor

namespace ai::cuda::shim {

// ---------- 메타: bias 크기 계산 ----------
[[nodiscard]] AI_INLINE std::size_t expected_bias_elems(int M, int N, BiasKind k) noexcept {
  switch (k) {
    case BiasKind::None:   return 0;
    case BiasKind::Scalar: return 1;
    case BiasKind::PerM:   return (M > 0) ? static_cast<std::size_t>(M) : 0;
    case BiasKind::PerN:   return (N > 0) ? static_cast<std::size_t>(N) : 0;
    default:               return 0;
  }
}

// ---------- 단일 원소 로드(Host/Device 호환) ----------
[[nodiscard]] AI_RD inline float ld1_compat(const float* p) noexcept {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  // 읽기 전용 캐시 힌트. 최신 아키텍처에선 일반 로드와 차이 미미하나 호환 유지.
  return __ldg(p);
#else
  return *p;
#endif
}

// ---------- 로우-레벨 포인터 기반 로드 ----------
[[nodiscard]] AI_RD inline float load_bias_ptr(const void* bias, BiasKind kind,
                                               int m, int n, int M, int N) noexcept {
  if (!bias || kind == BiasKind::None) return 0.f;

#ifdef AI_SHIM_DEBUG
  // 경계 방어(디버그 전용). 비용 최소화를 위해 assert 대신 빈 asm.
  if (kind == BiasKind::PerM) {
    if (!(m >= 0 && M > 0 && m < M)) asm volatile("");
  } else if (kind == BiasKind::PerN) {
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

// ---------- Params(POD) 기반 제네릭 로드 ----------
// P는 최소 다음 필드를 가져야 함: void* bias; BiasKind bias_kind; int M, N;
template <typename P>
[[nodiscard]] AI_RD inline float load_bias(const P& p, int m, int n) noexcept {
  return load_bias_ptr(p.bias, p.bias_kind, m, n, p.M, p.N);
}

template <typename P>
[[nodiscard]] AI_RD inline float add_bias(float z, const P& p, int m, int n) noexcept {
  return z + load_bias(p, m, n);
}

// ---------- 1D 텐서 길이로 BiasKind 추론 ----------
[[nodiscard]] AI_INLINE BiasKind
infer_bias_kind_1d_lenMN(const Tensor* Bias, std::int64_t M, std::int64_t N) noexcept {
  if (!Bias || !Bias->data)                 return BiasKind::None;
  if (Bias->desc.shape.size() != 1)         return BiasKind::None;
  const std::int64_t len = Bias->desc.shape[0];
  if (len == 1) return BiasKind::Scalar;
  if (len == N) return BiasKind::PerN;
  if (len == M) return BiasKind::PerM;
  return BiasKind::None;
}

[[nodiscard]] AI_INLINE BiasKind
deduce_bias_kind_from_forward(const Tensor* bias_like,
                              std::int64_t M, std::int64_t N) noexcept {
  return infer_bias_kind_1d_lenMN(bias_like, M, N);
}

} // namespace ai::cuda::shim
