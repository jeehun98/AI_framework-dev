// backends/cuda/ops/_common/shim/numeric.hpp

#pragma once
#include <cstdint>
#include <limits>
namespace ai::cuda::shim {
[[nodiscard]] inline bool fits_int32(std::int64_t x) noexcept {
  return x >= 0 && x <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
}
} // namespace ai::cuda::shim
