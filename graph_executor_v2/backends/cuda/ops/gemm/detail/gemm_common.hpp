// backends/cuda/ops/gemm/detail/gemm_common.hpp
#pragma once
#include <limits>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime_api.h>  // cudaMemcpyAsync, cudaStream_t

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "api.h"

namespace ai::gemm_common {

// ---------- RowMajor 2D 텐서의 leading dimension 추론 ----------
[[nodiscard]] inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) noexcept {
  // shape 검증
  if (t.desc.shape.size() != 2) return 0;
  // stride[0]가 존재하면 그대로 사용 (일부 런타임은 0을 "contiguous"로 쓰기도 함)
  if (t.desc.stride.size() >= 2) {
    const int64_t ld0 = t.desc.stride[0];
    if (ld0 > 0) return ld0;
    // ld0==0 또는 음수인 경우는 보호적으로 contiguous 가정
  }
  // stride 정보가 없으면 contiguous row-major 가정: ld = N
  return t.desc.shape[1];
}

// ---------- ai::ActKind → regemm::ActKind ----------
[[nodiscard]] inline regemm::ActKind to_regemm_act(ai::ActKind a) noexcept {
  using A = ai::ActKind; using R = regemm::ActKind;
  switch (a) {
    case A::None:      return R::None;
    case A::ReLU:      return R::ReLU;
    case A::LeakyReLU: return R::LeakyReLU;
    case A::GELU:      return R::GELU;
    case A::Sigmoid:   return R::Sigmoid;
    case A::Tanh:      return R::Tanh;
  }
  // 방어적 기본값
  return R::None;
}

// ---------- Bias 축 판정 (FWD 기준 규칙) ----------
// * len==1  → Scalar (최우선; N==1/M==1 케이스 보호)
// * len==N  → PerN (M==N이어도 PerN 우선)
// * len==M  → PerM
[[nodiscard]] inline regemm::BiasKind
infer_bias_kind_1d_lenMN(const ai::Tensor* Bias, int64_t M, int64_t N) noexcept {
  if (!Bias || !Bias->data)                return regemm::BiasKind::None;
  if (Bias->desc.shape.size() != 1)        return regemm::BiasKind::None;
  const int64_t len = Bias->desc.shape[0];
  if (len == 1) return regemm::BiasKind::Scalar;
  if (len == N) return regemm::BiasKind::PerN;
  if (len == M) return regemm::BiasKind::PerM;
  return regemm::BiasKind::None;
}

// ---------- FWD에서 사용한 bias 텐서로부터 BWD kind 재구성 ----------
[[nodiscard]] inline regemm::BiasKind
deduce_bias_kind_from_forward(const ai::Tensor* bias_like,
                              int64_t M, int64_t N) noexcept {
  return infer_bias_kind_1d_lenMN(bias_like, M, N);
}

// ---------- int64 → int32 범위 체크 ----------
[[nodiscard]] inline bool fits_int32(int64_t x) noexcept {
  // 행/열/ld는 음수 불가(하한 0), 상한은 INT_MAX
  return x >= 0 && x <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

// ---------- 공통 가드: 디바이스/타입/레이아웃 ----------
[[nodiscard]] inline bool is_cuda_f32_rowmajor(const ai::Tensor& T) noexcept {
  return T.is_cuda()
      && T.desc.dtype  == ai::DType::F32
      && T.desc.layout == ai::Layout::RowMajor
      && T.desc.shape.size() == 2; // 후속 코드 가정과 일치
}

// ---------- Z 버퍼 검증 (save_z용) ----------
[[nodiscard]] inline bool
validate_z_buffer(const ai::Tensor* Z, int64_t M, int64_t N) noexcept {
  if (!Z) return false;
  if (!is_cuda_f32_rowmajor(*Z)) return false;
  if (Z->desc.shape[0] != M || Z->desc.shape[1] != N) return false;
  const int64_t ldZ = infer_ld_rowmajor_2d(*Z);
  if (ldZ < N) return false;
  if (!fits_int32(ldZ)) return false;
  return true;
}

// ldZ를 함께 돌려받는 편의 오버로드
[[nodiscard]] inline bool
validate_z_buffer(const ai::Tensor* Z, int64_t M, int64_t N, int& out_ldZ) noexcept {
  if (!validate_z_buffer(Z, M, N)) return false;
  const int64_t ldZ64 = infer_ld_rowmajor_2d(*Z);
  out_ldZ = static_cast<int>(ldZ64);
  return true;
}

// ---------- D2D 복사 헬퍼 (act=None에서 Y<-Z 복사 등에 사용 가능) ----------
inline ai::Status
d2d_copy_async(void* dst, const void* src, size_t nbytes, ai::StreamHandle stream) noexcept {
  // 경계·특이 케이스: 0바이트, nullptr, 동일 포인터
  if (nbytes == 0 || dst == src) return ai::Status::Ok;
  if (!dst || !src)              return ai::Status::Invalid;

  auto s = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, s);
  return (err == cudaSuccess) ? ai::Status::Ok : ai::Status::RuntimeError;
}

} // namespace ai::gemm_common
