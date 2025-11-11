// backends/cuda/ops/_common/shim/epilogue_functors.hpp
#pragma once
#include "ai_defs.hpp"
#include "enums.hpp"        // ActKind, BiasKind
#include "activations.hpp"  // apply_act_runtime(...)
#include "bias.hpp"         // load_bias_ptr
#include <cstdint>

namespace ai::cuda::shim {

struct DropoutCtx {
  float p = 0.0f;                  // drop 확률 (keep = 1 - p)
  const uint8_t* mask = nullptr;   // [M*N] row-major (1=keep, 0=drop)
  float scale = 1.0f;              // 보정 스케일(보통 1/(1-p))
};

struct EpilogueCtx {
  // GEMM scalars
  float alpha = 1.f;
  float beta  = 0.f;

  // Bias/Act
  const void* bias = nullptr;
  BiasKind bias_kind = BiasKind::None;
  ActKind  act       = ActKind::None;
  float    leaky_slope = 0.01f;

  // (선택) C, Z 버퍼
  const float* C = nullptr; int ldc = 0;   // β*C용
  float* Z = nullptr;          int ldZ = 0; // pre-activation stash (save_preact==1일 때)
  int save_preact = 0;

  // (선택) Dropout
  DropoutCtx do_ctx{};
};

// 단일 스칼라 요소 처리 (m,n 위치)
AI_RD [[nodiscard]] inline float
apply_epilogue_scalar(float acc, int m, int n, int M, int N, const EpilogueCtx& e) noexcept
{
  // 1) pre = α·acc
  float pre = (e.alpha == 1.f) ? acc : (e.alpha * acc);

  // 2) + β·C
  const int ldc_eff = (e.ldc == 0 ? N : e.ldc);
  if (e.beta != 0.f && e.C) {
    const float cin = e.C[m * ldc_eff + n];
    pre = fmaf(e.beta, cin, pre);
  }

  // 3) + bias
  if (e.bias && e.bias_kind != BiasKind::None) {
    pre += load_bias_ptr(e.bias, e.bias_kind, m, n, M, N);
  }

  // 4) Z stash (옵션)
  if (e.save_preact && e.Z) {
    const int ldZ_eff = (e.ldZ == 0 ? N : e.ldZ);
    e.Z[m * ldZ_eff + n] = pre;
  }

  // 5) activation
  float out = apply_act_runtime(pre, e.act, e.leaky_slope);

  // 6) dropout (옵션)
  const bool use_do = (e.do_ctx.p > 0.f) && (e.do_ctx.mask != nullptr);
  if (use_do) {
    const uint8_t keep = e.do_ctx.mask[m * N + n];
    out = keep ? (out * e.do_ctx.scale) : 0.f;
  }
  return out;
}

// 벡터 처리: 커널 내에서 연속 n열에 대해 VEC개 요소를 한 번에 처리
template<int VEC>
AI_RD inline void apply_epilogue_vec(
    const float* __restrict__ acc_vec,
    float* __restrict__ out_vec,
    int m, int n0, int M, int N, const EpilogueCtx& e) noexcept
{
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const int n = n0 + i;
    const float v = apply_epilogue_scalar(acc_vec[i], m, n, M, N, e);
    out_vec[i] = v;
  }
}

} // namespace ai::cuda::shim
