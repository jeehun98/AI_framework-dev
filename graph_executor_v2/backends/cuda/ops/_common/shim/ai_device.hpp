// backends/cuda/ops/_common/shim/ai_device.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "ai_defs.hpp"  // for AI_INLINE

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Scalar kinds / layout
// ------------------------------------------------------------
enum class Device : int { CPU = 0, CUDA = 1 };
enum class DType  : int { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, I8 = 4 };
enum class Layout: int { RowMajor = 0, ColMajor = 1 };

// ---- ABI anchors (값 변경 시 즉시 감지) ----
static_assert(static_cast<int>(Device::CPU)  == 0, "Device ABI changed");
static_assert(static_cast<int>(Device::CUDA) == 1, "Device ABI changed");
static_assert(static_cast<int>(DType::F32)   == 0, "DType ABI changed");
static_assert(static_cast<int>(DType::F16)   == 1, "DType ABI changed");
static_assert(static_cast<int>(DType::BF16)  == 2, "DType ABI changed");
static_assert(static_cast<int>(DType::I32)   == 3, "DType ABI changed");
static_assert(static_cast<int>(DType::I8)    == 4, "DType ABI changed");
static_assert(static_cast<int>(Layout::RowMajor) == 0, "Layout ABI changed");
static_assert(static_cast<int>(Layout::ColMajor) == 1, "Layout ABI changed");

// ------------------------------------------------------------
// DType utils
// ------------------------------------------------------------
[[nodiscard]] AI_INLINE constexpr std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32:  return 4;
    case DType::F16:  return 2;
    case DType::BF16: return 2;
    case DType::I32:  return 4;
    case DType::I8:   return 1;
    default:          return 0;
  }
}

[[nodiscard]] AI_INLINE constexpr bool is_floating(DType dt) {
  return (dt == DType::F32) || (dt == DType::F16) || (dt == DType::BF16);
}

// ------------------------------------------------------------
// Safe arithmetic helpers (host-only small utils)
// ------------------------------------------------------------
[[nodiscard]] AI_INLINE constexpr std::int64_t safe_mul_nonneg(std::int64_t a, std::int64_t b) {
  // 음수 허용하지 않음, overflow 시 -1 반환
  if (a < 0 || b < 0) return -1;
  if (a == 0 || b == 0) return 0;
  // max / b < a  → overflow
  return (a > (std::numeric_limits<std::int64_t>::max() / b)) ? -1 : (a * b);
}

[[nodiscard]] AI_INLINE std::int64_t numel_of(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) return 0;
  std::int64_t n = 1;
  for (auto v : shape) {
    if (v < 0) return 0;
    n = safe_mul_nonneg(n, v);
    if (n < 0) return 0; // overflow
  }
  return n;
}

[[nodiscard]] AI_INLINE std::int64_t nbytes_of(const std::vector<std::int64_t>& shape, DType dt) {
  const auto n = numel_of(shape);
  if (n <= 0) return 0;
  const auto sz = static_cast<std::int64_t>(dtype_size(dt));
  return safe_mul_nonneg(n, sz); // overflow 시 -1
}

// ------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------
[[nodiscard]] AI_INLINE constexpr bool is_cuda(Device d)    { return d == Device::CUDA; }
[[nodiscard]] AI_INLINE constexpr bool is_rowmajor(Layout l){ return l == Layout::RowMajor; }

} // namespace ai::cuda::shim
