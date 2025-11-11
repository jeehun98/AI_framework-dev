// backends/cuda/ops/_common/shim/ai_tensor.hpp
#pragma once
#include <cstdint>
#include <type_traits>
#include <vector>
#include "ai_defs.hpp"
#include "ai_device.hpp"   // Device, DType, Layout, dtype_size, numel_of, nbytes_of, is_cuda/is_rowmajor
#include "layout.hpp"      // valid_ld_rowmajor, resolve_ld

namespace ai::cuda::shim {

// ---------------- Tensor (non-owning view) ----------------
struct TensorDesc {
  DType  dtype{DType::F32};
  Layout layout{Layout::RowMajor};
  std::vector<std::int64_t> shape;   // dims
  std::vector<std::int64_t> stride;  // element strides

  [[nodiscard]] AI_INLINE std::int64_t dim(int i) const {
    return shape.at(static_cast<std::size_t>(i));
  }
};

struct Tensor {
  void*      data{nullptr};      // device pointer (CUDA)
  TensorDesc desc{};
  Device     device{Device::CUDA};
  int        device_index{0};

  // -------- basic checks --------
  [[nodiscard]] AI_INLINE bool is_cuda()    const { return device == Device::CUDA; }
  [[nodiscard]] AI_INLINE bool is_defined() const { return data != nullptr; }

  // -------- layout/contiguity --------
  [[nodiscard]] AI_INLINE bool is_contiguous_rowmajor_2d() const {
    if (desc.layout != Layout::RowMajor) return false;
    if (desc.shape.size() != 2 || desc.stride.size() != 2) return false;
    const std::int64_t rows = desc.shape[0];
    const std::int64_t cols = desc.shape[1];
    if (rows < 0 || cols < 0) return false;
    // row-major contiguous: stride = {cols, 1}
    return (desc.stride[1] == 1) && (desc.stride[0] == cols);
  }

  // -------- sizes --------
  [[nodiscard]] AI_INLINE std::int64_t numel()  const { return numel_of(desc.shape); }
  [[nodiscard]] AI_INLINE std::int64_t nbytes() const { return nbytes_of(desc.shape, desc.dtype); }

  // -------- raw pointers --------
  AI_INLINE void*       data_ptr()       { return data; }
  AI_INLINE const void* data_ptr() const { return data; }

  // -------- typed pointers --------
  template <typename T>
  AI_INLINE T* data_ptr() {
    static_assert(!std::is_const<T>::value, "Use const overload for const type");
    return reinterpret_cast<T*>(data);
  }
  template <typename T>
  AI_INLINE const T* data_ptr() const {
    return reinterpret_cast<const T*>(data);
  }
};

// ---------------- helpers ----------------

// leading dimension helpers (return -1 if not 2D)
[[nodiscard]] AI_INLINE std::int64_t lda(const Tensor& A) {
  if (A.desc.stride.size() != 2) return -1;
  return (A.desc.layout == Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
[[nodiscard]] AI_INLINE std::int64_t ldb(const Tensor& B) {
  if (B.desc.stride.size() != 2) return -1;
  return (B.desc.layout == Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
[[nodiscard]] AI_INLINE std::int64_t ldd(const Tensor& D) {
  if (D.desc.stride.size() != 2) return -1;
  return (D.desc.layout == Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

[[nodiscard]] AI_INLINE std::vector<std::int64_t>
make_rowmajor_strides(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> s(shape.size());
  std::int64_t st = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    s[static_cast<std::size_t>(i)] = st;
    st *= shape[static_cast<std::size_t>(i)];
  }
  return s;
}

// factories (non-owning views)
[[nodiscard]] AI_INLINE Tensor make_tensor2d(void* ptr, std::int64_t rows, std::int64_t cols,
                                             DType dt = DType::F32) {
  Tensor t;
  t.data = ptr; t.device = Device::CUDA; t.device_index = 0;
  t.desc.dtype = dt; t.desc.layout = Layout::RowMajor;
  t.desc.shape = {rows, cols}; t.desc.stride = {cols, 1};
  return t;
}

[[nodiscard]] AI_INLINE Tensor make_tensor_from_ptr(void* ptr,
                                                    const std::vector<std::int64_t>& shape,
                                                    DType dt = DType::F32,
                                                    Layout layout = Layout::RowMajor) {
  Tensor t;
  t.data = ptr; t.device = Device::CUDA; t.device_index = 0;
  t.desc.dtype = dt; t.desc.layout = layout;
  t.desc.shape = shape;
  t.desc.stride = (layout == Layout::RowMajor)
                    ? make_rowmajor_strides(shape)
                    : std::vector<std::int64_t>{}; // 필요 시 col-major 추가
  return t;
}

} // namespace ai::cuda::shim
