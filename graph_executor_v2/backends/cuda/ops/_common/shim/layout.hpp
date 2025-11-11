// backends/cuda/ops/_common/shim/layout.hpp
#pragma once
#include <cstddef>
#include "ai_defs.hpp"  // for AI_INLINE

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Layout utilities (Row-major 전제의 leading dimension 검증)
// ------------------------------------------------------------

/// Row-major 배열에서 ld(leading dimension)가 유효한지 확인
[[nodiscard]] AI_INLINE bool valid_ld_rowmajor(int rows, int cols, int ld) noexcept {
  if (rows <= 0 || cols <= 0) return false;
  if (ld == 0) return true;   // 0이면 후단에서 cols로 해석 (연속 메모리)
  return ld >= cols;
}

/// 0인 ld를 fallback_cols로 대체
[[nodiscard]] AI_INLINE int resolve_ld(int ld, int fallback_cols) noexcept {
  return (ld == 0) ? fallback_cols : ld;
}

[[nodiscard]] AI_INLINE bool contiguous_rowmajor(int rows, int cols, int ld) noexcept {
  return ld == cols || ld == 0;
}

[[nodiscard]] AI_INLINE std::size_t offset_2d(std::size_t row, std::size_t col, std::size_t ld) noexcept {
  return row * static_cast<std::size_t>(ld) + col;
}


} // namespace ai::cuda::shim
