#pragma once
#include "regemm/api.h"

namespace regemm {

// Expected layouts (host side responsibility):
//  - Scalar: bias points to single float
//  - PerM  : bias points to float[M]
//  - PerN  : bias points to float[N]
__device__ __forceinline__
float load_bias(const GemmBiasActParams& p, int m, int n) {
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
