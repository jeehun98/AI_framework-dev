#pragma once
#include "ai_defs.hpp"
#include "ai_tensor.hpp"
#include "ai_validate.hpp"
#include <limits>
#include <cstdint>

namespace ai::cuda::shim {

// RowMajor 2D의 유효 ld 추론 (stride[0] 우선, 없으면 N)
[[nodiscard]] inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) noexcept {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) {
    const int64_t ld0 = t.desc.stride[0];
    if (ld0 > 0) return ld0;
  }
  return t.desc.shape[1];
}

// Z 버퍼 검증 (ldZ 함께 반환)
[[nodiscard]] inline bool
validate_z_buffer(const ai::Tensor* Z, int64_t M, int64_t N, int& out_ldZ) noexcept {
  if (!Z) return false;
  if (!ai::cuda::shim::is_cuda_f32_rowmajor(*Z)) return false;
  if (Z->desc.shape[0] != M || Z->desc.shape[1] != N) return false;
  const int64_t ldZ64 = infer_ld_rowmajor_2d(*Z);
  if (ldZ64 < N) return false;
  if (ldZ64 < 0 || ldZ64 > static_cast<int64_t>(std::numeric_limits<int>::max())) return false;
  out_ldZ = static_cast<int>(ldZ64);
  return true;
}

} // namespace ai::cuda::shim
