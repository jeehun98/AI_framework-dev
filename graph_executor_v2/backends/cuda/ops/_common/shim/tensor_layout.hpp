// backends/cuda/ops/_common/shim/tensor_layout.hpp
#pragma once
#include <cstdint>
#include "ai_defs.hpp"      // AI_INLINE
#include "ai_tensor.hpp"    // Tensor
#include "ai_validate.hpp"  // is_cuda_f32_rowmajor()
#include "numeric.hpp"      // fits_int32()

namespace ai::cuda::shim {

// Row-major 2D: stride[0]가 양수면 LD로, 아니면 N 사용
[[nodiscard]] AI_INLINE std::int64_t infer_ld_rowmajor_2d(const Tensor& t) noexcept {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) {
    const std::int64_t ld0 = t.desc.stride[0];
    if (ld0 > 0) return ld0;
  }
  return static_cast<std::int64_t>(t.desc.shape[1]);
}

// Z 버퍼 검증 + ldZ 산출 (true=유효)
[[nodiscard]] AI_INLINE bool
validate_z_buffer(const Tensor* Z, std::int64_t M, std::int64_t N, int& out_ldZ) noexcept {
  if (!Z) return false;
  if (!is_cuda_f32_rowmajor(*Z)) return false;           // 디바이스/레이아웃/dtype/랭크 검사
  if (Z->desc.shape.size() != 2) return false;
  if (Z->desc.shape[0] != M || Z->desc.shape[1] != N) return false;

  const std::int64_t ldZ64 = infer_ld_rowmajor_2d(*Z);
  if (ldZ64 < N) return false;                           // Row-major 규칙 위반
  if (!fits_int32(ldZ64)) return false;                  // int 변환 안전성

  out_ldZ = static_cast<int>(ldZ64);
  return true;
}

} // namespace ai::cuda::shim
