#pragma once
#include <limits>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

#include "regemm/api.h"

namespace ai::gemm_common {

// -------- RowMajor 2D 텐서의 leading dimension 추론 --------
inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1]; // contiguous row-major 가정
}

// -------- ai::ActKind → regemm::ActKind --------
inline regemm::ActKind to_regemm_act(ai::ActKind a) {
  using A = ai::ActKind; using R = regemm::ActKind;
  switch (a) {
    case A::None:      return R::None;
    case A::ReLU:      return R::ReLU;
    case A::LeakyReLU: return R::LeakyReLU;
    case A::GELU:      return R::GELU;
    case A::Sigmoid:   return R::Sigmoid;
    case A::Tanh:      return R::Tanh;
  }
  return R::None;
}

// -------- Bias 축 판정(FWD 기준 규칙) --------
// * 길이 1 ⇒ Scalar (항상 최우선; N==1/M==1 보호)
// * 길이==N ⇒ PerN (M==N이어도 PerN 우선)
// * 길이==M ⇒ PerM
inline regemm::BiasKind infer_bias_kind_1d_lenMN(const ai::Tensor* Bias, int64_t M, int64_t N) {
  if (!Bias || !Bias->data) return regemm::BiasKind::None;
  if (Bias->desc.shape.size() != 1) return regemm::BiasKind::None;
  const int64_t len = Bias->desc.shape[0];
  if (len == 1) return regemm::BiasKind::Scalar;
  if (len == N) return regemm::BiasKind::PerN;
  if (len == M) return regemm::BiasKind::PerM;
  return regemm::BiasKind::None;
}

// -------- FWD에서 사용한 bias 텐서로부터 BWD용 kind 재구성 --------
inline regemm::BiasKind deduce_bias_kind_from_forward(const ai::Tensor* bias_like,
                                                      int64_t M, int64_t N) {
  return infer_bias_kind_1d_lenMN(bias_like, M, N);
}

// -------- int64 → int32 범위 체크 --------
inline bool fits_int32(int64_t x) {
  return x >= std::numeric_limits<int>::min() && x <= std::numeric_limits<int>::max();
}

// -------- 공통 가드(디바이스/타입/레이아웃/FWD-전제) --------
inline bool is_cuda_f32_rowmajor(const ai::Tensor& T) {
  return T.is_cuda() && T.desc.dtype==ai::DType::F32 && T.desc.layout==ai::Layout::RowMajor;
}

} // namespace ai::gemm_common
