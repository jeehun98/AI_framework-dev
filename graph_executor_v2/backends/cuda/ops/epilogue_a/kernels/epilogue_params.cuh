// backends/cuda/ops/epilogue/kernels/epilogue_params.cuh
#pragma once
#include <cstdint>
#include "../api/epilogue.h"
#include "../api/dtype.h"
#include <cuda_fp16.h>

namespace epi {

template <typename T>
struct EpParams {
  const T* __restrict__ x;     // [M,N]
  const T* __restrict__ bias;  // [N] or nullptr
  T* __restrict__ y;           // [M,N]
  uint8_t* __restrict__ mask;  // [M,N] or nullptr

  // (옵션) residual blend에 대비한 포인터
  const T* __restrict__ resid = nullptr;

  int64_t M, N;
  int64_t ld_x, ld_y, ld_bias;

  // dropout
  float dropout_p;         // drop prob in [0,1)
  uint64_t seed;
  bool save_mask;

  // === ep_apply 호환용 확장 필드 ===
  // p_drop/keep_scale: DropF.apply에 전달
  float p_drop = 0.0f;     // == dropout_p
  float keep_scale = 1.0f; // (dropout_p>0 ? 1/(1-p) : 1)
  // BlendF용
  float alpha = 1.0f;
  float beta  = 0.0f;

  __device__ __forceinline__ int64_t idx(int64_t r, int64_t c) const { return r*ld_x + c; }
};

template <typename T>
__host__ inline EpParams<T> make_params(const Plan& plan, const Tensors& t) {
  EpParams<T> p{};
  p.x       = reinterpret_cast<const T*>(t.x);
  p.bias    = reinterpret_cast<const T*>(t.bias);
  p.y       = reinterpret_cast<T*>(t.y);
  p.mask    = reinterpret_cast<uint8_t*>(t.mask_out);
  p.M       = plan.rows;
  p.N       = plan.cols;
  p.ld_x    = plan.ld_x ? plan.ld_x : plan.cols;
  p.ld_y    = plan.ld_y ? plan.ld_y : plan.cols;
  p.ld_bias = plan.ld_bias ? plan.ld_bias : plan.cols;

  p.dropout_p = plan.attrs.dropout_p;
  p.seed      = plan.attrs.seed;
  p.save_mask = plan.attrs.save_mask;

  // === 호환 필드 초기화 ===
  p.p_drop = p.dropout_p;
  p.keep_scale = (p.dropout_p > 0.f && p.dropout_p < 1.f) ? (1.f / (1.f - p.dropout_p)) : 1.f;
  p.alpha = 1.0f;
  p.beta  = 0.0f;
  p.resid = nullptr; // 필요 시 외부에서 세팅

  return p;
}

// === ep_apply가 기대하는 alias 제공 ===
using EpParamsF32 = EpParams<float>;
using EpParamsF16 = EpParams<half>;

} // namespace epi
