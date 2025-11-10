#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <limits>

namespace ai {

enum class Device { CPU, CUDA };
enum class DType  { F32, F16, BF16, I32, I8 };
enum class Layout { RowMajor, ColMajor };

// ---- dtype size helper ----
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

// ---- safe multiply / numel (MSVC-friendly) ----
inline bool mul_overflow_u64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  if (a == 0 || b == 0) { out = 0; return false; }
  if (a > std::numeric_limits<std::uint64_t>::max() / b) return true;
  out = a * b;
  return false;
}

inline std::int64_t numel_safe(const std::vector<std::int64_t>& shape){
  if (shape.empty()) return 0;
  std::uint64_t acc = 1;
  for (auto v : shape) {
    if (v < 0) return 0; // 음수 차원 비허용
    std::uint64_t tmp;
    if (mul_overflow_u64(acc, static_cast<std::uint64_t>(v), tmp)) return 0;
    acc = tmp;
    if (acc > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return 0;
  }
  return static_cast<std::int64_t>(acc);
}

// ---- tensor types ----
struct TensorDesc {
  DType  dtype  { DType::F32 };
  Layout layout { Layout::RowMajor };
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
  std::int64_t dim(int i) const { return shape.at(static_cast<std::size_t>(i)); }
};

struct Tensor {
  void*   data{nullptr};
  TensorDesc desc{};
  Device  device{Device::CUDA};
  int     device_index{0};

  // predicates
  bool is_defined() const { return data != nullptr; }
  bool is_cuda()    const { return device == Device::CUDA; }

  bool is_contiguous_rowmajor_2d() const {
    if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
    if (desc.stride.size() != 2) return false;
    const std::int64_t rows = desc.shape[0];
    const std::int64_t cols = desc.shape[1];
    return (desc.stride[1] == 1) && (desc.stride[0] == cols) && (rows >= 0 && cols >= 0);
  }

  // sizes
  std::int64_t numel() const { return numel_safe(desc.shape); }
  std::int64_t nbytes() const {
    return static_cast<std::int64_t>(numel()) * static_cast<std::int64_t>(dtype_size(desc.dtype));
  }

  // raw / typed pointers
  void*       data_ptr()       { return data; }
  const void* data_ptr() const { return data; }

  template <typename T>
  T* data_ptr() {
    static_assert(!std::is_const<T>::value, "Use const overload for const type");
    return reinterpret_cast<T*>(data);
  }
  template <typename T>
  const T* data_ptr() const {
    return reinterpret_cast<const T*>(data);
  }
};

// ---- GEMM helpers (2D leading dims) ----
inline std::int64_t lda(const Tensor& A){
  if (A.desc.stride.size()!=2) return -1;
  return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
inline std::int64_t ldb(const Tensor& B){
  if (B.desc.stride.size()!=2) return -1;
  return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
inline std::int64_t ldd(const Tensor& D){
  if (D.desc.stride.size()!=2) return -1;
  return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

} // namespace ai
