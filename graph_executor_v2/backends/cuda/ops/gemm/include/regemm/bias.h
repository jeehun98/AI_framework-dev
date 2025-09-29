#pragma once
#include "regemm/api.h"


namespace regemm {

// Host 측 레이아웃 규약
//  - Scalar: bias -> float* (1개)
//  - PerM  : bias -> float* (M개)
//  - PerN  : bias -> float* (N개)

// NOTE:
// GemmBiasActParams / GemmBiasActParamsEx 모두
//  - p.bias (const void*)
//  - p.bias_kind (BiasKind)
//  - p.M, p.N  (dims; PerM/PerN 인덱싱 검사용) 를 동일 이름으로 갖는다.
// 그래서 템플릿으로 공용 처리 가능.

template <typename P>
__device__ __forceinline__
float load_bias(const P& p, int m, int n) {
  if (!p.bias || p.bias_kind == BiasKind::None) return 0.f;

  const float* b = reinterpret_cast<const float*>(p.bias);
  switch (p.bias_kind) {
    case BiasKind::Scalar: return *b;
    case BiasKind::PerM:   return b[m];
    case BiasKind::PerN:   return b[n];
    default:               return 0.f;
  }
}

} // namespace regemm
