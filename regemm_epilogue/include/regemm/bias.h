#pragma once
#include "api.h"
namespace regemm {

__device__ __forceinline__ float load_bias(
    const GemmBiasActParams& p, int m, int n) {
  if (!p.bias) return 0.f;
  const float* b = reinterpret_cast<const float*>(p.bias);
  switch (p.bias_kind) {
    case BiasKind::PerN: return b[n];
    case BiasKind::PerM: return b[m];
    case BiasKind::Scalar: return b[0];
    default: return 0.f;
  }
}

} // namespace regemm
