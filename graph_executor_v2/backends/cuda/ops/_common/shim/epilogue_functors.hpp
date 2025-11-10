// backends/cuda/ops/_common/shim/epilogue_functors.hpp

#pragma once
#include "ai_defs.hpp"
#include "enums.hpp"        // ActKind, BiasKind
#include "activations.hpp"  // apply_act_runtime(...)
#include "bias.hpp"         // load_bias/load_bias_ptr
#include <cstdint>

namespace ai::cuda::shim {

// 외부에서 제공 가능한 dropout 구성
struct DropoutCtx {
  float p = 0.0f;                // keep-prob = 1-p
  const uint8_t* mask = nullptr; // [M*N] or vectorized; 1=keep,0=drop
  float scale = 1.0f;            // 보정 스케일(일반적으로 1/(1-p))
  // RNG 방식(Philox 등)을 쓰고 싶다면 별도 rng ctx를 정의해 오버로드하면 됨.
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
  float* Z = nullptr;          int ldZ = 0; // pre-activation stash
  int save_preact = 0;                     // 1이면 Z에 쓰기

  // (선택) Dropout
  DropoutCtx do_ctx{};
};

// 단일 스칼라 요소 처리 (m,n 위치)
AI_RD inline float apply_epilogue_scalar(
    float acc, int m, int n, int M, int N, const EpilogueCtx& e)
{
  // 1) pre = α·acc + β·C + bias
  float pre = e.alpha * acc;

  if (e.beta != 0.f && e.C) {
    const float* cptr = e.C + m * (e.ldc ? e.ldc : N) + n;
    pre = fmaf(e.beta, *cptr, pre);
  }
  pre = pre + load_bias_ptr(e.bias, e.bias_kind, m, n, M, N);

  // (옵션) Z stash
  if (e.save_preact && e.Z) {
    float* zptr = e.Z + m * (e.ldZ ? e.ldZ : N) + n;
    *zptr = pre;
  }

  // 2) act
  float out = apply_act_runtime(pre, e.act, e.leaky_slope);

  // 3) dropout (외부 mask 또는 없음)
  if (e.do_ctx.p > 0.f && e.do_ctx.mask) {
    const uint8_t keep = e.do_ctx.mask[m * N + n];
    out = keep ? (out * e.do_ctx.scale) : 0.f;
  }

  return out;
}

// 벡터화 버전의 인터페이스(예시): 커널에서 포인터/stride만 넘겨 batch 적용
template<int VEC>
AI_RD inline void apply_epilogue_vec(
    const float* acc_vec, float* out_vec,
    int m, int n0, int M, int N, const EpilogueCtx& e)
{
  #pragma unroll
  for (int i = 0; i < VEC; ++i) {
    int n = n0 + i;
    float v = apply_epilogue_scalar(acc_vec[i], m, n, M, N, e);
    out_vec[i] = v;
  }
}

} // namespace ai::cuda::shim
