#pragma once
#include "../api/epilogue.h"

namespace epi {
inline const char* validate_common(const Plan& plan, const Tensors& ts){
  if (ts.M<=0 || ts.N<=0) return "M,N must be positive";
  if (!ts.x || !ts.y)     return "x,y required";
  if (plan.attrs.bias==BiasKind::PerN && !ts.bias) return "bias==nullptr";
  if (plan.attrs.dropout && ts.rng_seed==0ULL) return "dropout needs rng_seed!=0";
  return nullptr;
}
} // namespace epi
