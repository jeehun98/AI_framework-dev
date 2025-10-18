#pragma once
#include <cuda_runtime.h>
#include "../api/epilogue.h"
#include "../kernels/epilogue_params.cuh"

namespace epi {

inline dim3 make_grid(int total, int block=256) {
  int g = (total + block - 1) / block;
  if (g <= 0) g = 1;
  return dim3(g);
}

inline EpParamsF32 pack_fp32(const Plan& plan, const Tensors& ts){
  EpParamsF32 P{};
  P.M=ts.M; P.N=ts.N;
  P.ld_x = ts.ld_x? ts.ld_x: ts.N;
  P.ld_y = ts.ld_y? ts.ld_y: ts.N;
  P.x = static_cast<const float*>(ts.x);
  P.y = static_cast<float*>(ts.y);
  P.bias = static_cast<const float*>(ts.bias);
  P.alpha = plan.attrs.alpha; P.beta = plan.attrs.beta;
  P.act = (plan.attrs.act==ActKind::ReLU)?1:0;
  P.has_bias = (plan.attrs.bias==BiasKind::PerN)?1:0;
  P.use_dropout = plan.attrs.dropout?1:0;
  P.p_drop = plan.attrs.p_drop;
  P.keep_scale = (plan.attrs.dropout && plan.attrs.p_drop<1.f) ? 1.f/(1.f-plan.attrs.p_drop) : 1.f;
  P.seed = ts.rng_seed; P.offset = ts.rng_offset;
  return P;
}

inline EpParamsF16 pack_fp16(const Plan& plan, const Tensors& ts){
  EpParamsF16 P{};
  P.M=ts.M; P.N=ts.N;
  P.ld_x = ts.ld_x? ts.ld_x: ts.N;
  P.ld_y = ts.ld_y? ts.ld_y: ts.N;
  P.x = static_cast<const half*>(ts.x);
  P.y = static_cast<half*>(ts.y);
  P.bias = static_cast<const half*>(ts.bias);
  P.alpha = plan.attrs.alpha; P.beta = plan.attrs.beta;
  P.act = (plan.attrs.act==ActKind::ReLU)?1:0;
  P.has_bias = (plan.attrs.bias==BiasKind::PerN)?1:0;
  P.use_dropout = plan.attrs.dropout?1:0;
  P.p_drop = plan.attrs.p_drop;
  P.keep_scale = (plan.attrs.dropout && plan.attrs.p_drop<1.f) ? 1.f/(1.f-plan.attrs.p_drop) : 1.f;
  P.seed = ts.rng_seed; P.offset = ts.rng_offset;
  return P;
}

} // namespace epi
