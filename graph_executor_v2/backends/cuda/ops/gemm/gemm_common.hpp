#pragma once
#include <limits>
#include <cstdint>
#include <cuda_runtime.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "regemm/api.h"

namespace ai::gemm_common {

// ---------- RowMajor 2D 텐서의 leading dimension 추론 ----------
inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  // shape 검증
  if (t.desc.shape.size() != 2) return 0;
  // stride[0]가 존재하면 그대로 사용 (row-major일 때 보통 stride[0] >= N)
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  // stride 정보가 없으면 contiguous row-major 가정: ld = N
  return t.desc.shape[1];
}

// ---------- ai::ActKind → regemm::ActKind ----------
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

// ---------- Bias 축 판정 (FWD 기준 규칙) ----------
// * len==1  → Scalar (최우선; N==1/M==1 케이스 보호)
// * len==N  → PerN (M==N이어도 PerN 우선)
// * len==M  → PerM
inline regemm::BiasKind infer_bias_kind_1d_lenMN(const ai::Tensor* Bias, int64_t M, int64_t N) {
  if (!Bias || !Bias->data) return regemm::BiasKind::None;
  if (Bias->desc.shape.size() != 1) return regemm::BiasKind::None;
  const int64_t len = Bias->desc.shape[0];
  if (len == 1) return regemm::BiasKind::Scalar;
  if (len == N) return regemm::BiasKind::PerN;
  if (len == M) return regemm::BiasKind::PerM;
  return regemm::BiasKind::None;
}

// ---------- FWD에서 사용한 bias 텐서로부터 BWD kind 재구성 ----------
inline regemm::BiasKind deduce_bias_kind_from_forward(const ai::Tensor* bias_like,
                                                      int64_t M, int64_t N) {
  return infer_bias_kind_1d_lenMN(bias_like, M, N);
}

// ---------- int64 → int32 범위 체크 ----------
inline bool fits_int32(int64_t x) {
  // 행/열/ld 같은 크기/스트라이드는 음수가 올 수 없으므로 하한 0으로 제한해도 안전
  return x >= 0 && x <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

// ---------- 공통 가드: 디바이스/타입/레이아웃 ----------
inline bool is_cuda_f32_rowmajor(const ai::Tensor& T) {
  return T.is_cuda()
      && T.desc.dtype  == ai::DType::F32
      && T.desc.layout == ai::Layout::RowMajor;
}

// ---------- Z 버퍼 검증 (save_z용) ----------
inline bool validate_z_buffer(const ai::Tensor* Z, int64_t M, int64_t N) {
  if (!Z) return false;
  if (!is_cuda_f32_rowmajor(*Z)) return false;
  if (Z->desc.shape.size() != 2) return false;
  if (Z->desc.shape[0] != M || Z->desc.shape[1] != N) return false;
  const int64_t ldZ = infer_ld_rowmajor_2d(*Z);
  if (ldZ < N) return false;
  if (!fits_int32(ldZ)) return false;
  return true;
}

// ---------- D2D 복사 헬퍼 (act=None에서 Y<-Z 복사 등에 사용 가능) ----------
inline ai::Status d2d_copy_async(void* dst, const void* src, size_t nbytes, ai::StreamHandle stream) {
  auto s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, s);
  return (err == cudaSuccess) ? ai::Status::Ok : ai::Status::RuntimeError;
}

} // namespace ai::gemm_common
