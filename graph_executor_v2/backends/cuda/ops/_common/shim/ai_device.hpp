#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ai {

// ---------------- Scalar / layout ----------------
enum class Device { CPU, CUDA };
enum class DType  { F32, F16, BF16, I32, I8 };
enum class Layout { RowMajor, ColMajor };

[[nodiscard]] inline constexpr std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32:  return 4;
    case DType::F16:  return 2;
    case DType::BF16: return 2;
    case DType::I32:  return 4;
    case DType::I8:   return 1;
    default:          return 0;
  }
}

inline int64_t safe_mul_nonneg(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return 0;
  if (a > (std::numeric_limits<int64_t>::max() / b)) return -1;
  return a * b;
}
inline int64_t numel_of(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 0;
  int64_t n = 1;
  for (auto v : shape) {
    if (v < 0) return 0;
    n = safe_mul_nonneg(n, v);
    if (n < 0) return 0;
  }
  return n;
}

} // namespace ai
