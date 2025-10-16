// backends/cuda/ops/gemm/detail/bias.h
#pragma once
#include "api.h"

#include <cstdint>
#ifdef REGEMM_DEBUG
  #include <cassert>
#endif

// 호스트 단위테스트를 원하면 컴파일 옵션에 -DREGEMM_TEST_ON_HOST 추가
#ifdef REGEMM_TEST_ON_HOST
  #define RD __host__ __device__
#else
  #define RD __device__
#endif

namespace regemm {

// Host 측 레이아웃 규약
//  - Scalar: bias -> float* (1개)
//  - PerM  : bias -> float* (M개)
//  - PerN  : bias -> float* (N개)
//
// NOTE:
// GemmBiasActParams / GemmBiasActParamsEx 공통 필드
//   - p.bias (const void*)
//   - p.bias_kind (BiasKind)
//   - p.M, p.N     (PerM/PerN 인덱싱 검사용)
// 을 동일 이름으로 갖기 때문에 템플릿으로 공용 처리 가능.

// 디바이스에서 읽기 힌트(RO 캐시) + 호스트 호환
RD __forceinline__ float ld1_compat(const float* p) {
#if defined(__CUDA_ARCH__)
  // __ldg는 const 포인터에서 읽기 캐시 힌트를 줌
  return __ldg(p);
#else
  return *p;
#endif
}

template <typename P>
RD __forceinline__
float load_bias(const P& p, int m, int n) {
  if (!p.bias || p.bias_kind == BiasKind::None) return 0.f;

#ifdef REGEMM_DEBUG
  // 음수 인덱스 및 치수 검증(디버그 빌드에서만)
  if (p.bias_kind == BiasKind::PerM) {
    assert(m >= 0 && p.M > 0 && m < p.M && "PerM bias index out of range");
  } else if (p.bias_kind == BiasKind::PerN) {
    assert(n >= 0 && p.N > 0 && n < p.N && "PerN bias index out of range");
  }
#endif

  const float* b = reinterpret_cast<const float*>(p.bias);
  switch (p.bias_kind) {
    case BiasKind::Scalar:
      return ld1_compat(b);
    case BiasKind::PerM:
      return ld1_compat(b + m);
    case BiasKind::PerN:
      return ld1_compat(b + n);
    case BiasKind::None:
    default:
      return 0.f;
  }
}

// (옵션) 누산값에 bias를 더한 값을 반환하는 헬퍼
template <typename P>
RD __forceinline__
float add_bias(float z, const P& p, int m, int n) {
  return z + load_bias(p, m, n);
}

} // namespace regemm

#ifdef REGEMM_TEST_ON_HOST
#undef RD
#endif
