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

inline const char* validate_ffn_gelu_dropout_resid(const Plan& plan, const Tensors& ts){
  if (ts.M<=0 || ts.N<=0) return "M,N must be positive";
  if (!ts.x || !ts.y)     return "x,y required";
  if (plan.attrs.bias==BiasKind::PerN && !ts.bias) return "bias==nullptr";
  if (!plan.attrs.dropout) return "recipe requires dropout=true";
  if (ts.rng_seed==0ULL)  return "dropout requires rng_seed!=0";
  return nullptr;
}

inline const char* validate_cnn_bias_relu(const Plan& plan, const Tensors& ts){
  if (ts.M<=0 || ts.N<=0) return "M,N must be positive";
  if (!ts.x || !ts.y)     return "x,y required";
  if (plan.attrs.bias!=BiasKind::PerN) return "CNN recipe requires PerN bias";
  if (!ts.bias) return "bias==nullptr";
  return nullptr;
}

inline const char* validate_gru3(const Plan& plan, const Tensors& ts){
  if (ts.M<=0 || ts.N<=0) return "M,N must be positive";
  if (!ts.x || !ts.y)     return "x,y required";
  if (!ts.bias)           return "bias==nullptr (expect 3N bias)";
  if (!ts.resid)          return "resid==nullptr (need h_prev)";
  // ld_x는 3N이어야 안정. 상위에서 ts.ld_x = 3N으로 맞춰서 전달 권장.
  return nullptr;
}
} // namespace epi
