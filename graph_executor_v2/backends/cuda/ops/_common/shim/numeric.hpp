// backends/cuda/ops/_common/shim/numeric.hpp
#pragma once
#include <cstdint>
#include <limits>
#include "ai_defs.hpp"  // for AI_INLINE

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Numeric utils (간단한 정수 범위 검사 등)
// ------------------------------------------------------------

/// 64비트 값이 int32_t 범위에 들어가는지 확인
[[nodiscard]] AI_INLINE bool fits_int32(std::int64_t x) noexcept {
  return x >= 0 &&
         x <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
}

[[nodiscard]] AI_INLINE constexpr bool is_pow2(std::uint64_t x) noexcept {
  return x && !(x & (x - 1));
}

[[nodiscard]] AI_INLINE constexpr std::uint32_t div_up(std::uint32_t a, std::uint32_t b) noexcept {
  return (a + b - 1) / b;
}

[[nodiscard]] AI_INLINE constexpr std::size_t align_up(std::size_t x, std::size_t alignment) noexcept {
  return ((x + alignment - 1) / alignment) * alignment;
}


} // namespace ai::cuda::shim
