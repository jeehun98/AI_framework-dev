// backends/cuda/ops/_common/shim/ai_tensor.hpp
#pragma once
#include <cstdint>
#include <type_traits>
#include <vector>
#include "ai_device.hpp"

namespace ai {

// ---------------- Tensor (비소유 뷰) ----------------
struct TensorDesc {
  DType dtype{DType::F32};
  Layout layout{Layout::RowMajor};
  std::vector<int64_t> shape;   // dims
  std::vector<int64_t> stride;  // element strides
  int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
};

struct Tensor {
  void*  data{nullptr};     // device ptr (CUDA)
  TensorDesc desc{};
  Device device{Device::CUDA};
  int    device_index{0};

  bool is_cuda()     const { return device == Device::CUDA; }
  bool is_defined()  const { return data != nullptr; }

  bool is_contiguous_rowmajor_2d() const {
    if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
    if (desc.stride.size() != 2) return false;
    const int64_t rows = desc.shape[0];
    const int64_t cols = desc.shape[1];
    return (desc.stride[1] == 1) && (desc.stride[0] == cols) && (rows >= 0 && cols >= 0);
  }

  int64_t numel()  const { return numel_of(desc.shape); }
  int64_t nbytes() const { return static_cast<int64_t>(numel()) * static_cast<int64_t>(dtype_size(desc.dtype)); }

  // Raw
  void* data_ptr() { return data; }
  const void* data_ptr() const { return data; }

  // Typed
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

namespace shim {

// leading dimension helpers
inline int64_t lda(const Tensor& A){
  if (A.desc.stride.size() != 2) return -1;
  return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
inline int64_t ldb(const Tensor& B){
  if (B.desc.stride.size() != 2) return -1;
  return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
inline int64_t ldd(const Tensor& D){
  if (D.desc.stride.size() != 2) return -1;
  return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

inline std::vector<int64_t> make_rowmajor_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size());
  int64_t st = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    s[i] = st; st *= shape[i];
  }
  return s;
}
inline Tensor make_tensor2d(void* ptr, int64_t rows, int64_t cols) {
  Tensor t;
  t.data = ptr; t.device = Device::CUDA; t.device_index = 0;
  t.desc.dtype = DType::F32; t.desc.layout = Layout::RowMajor;
  t.desc.shape = {rows, cols}; t.desc.stride = {cols, 1};
  return t;
}
inline Tensor make_tensor_from_ptr(void* ptr, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = ptr; t.device = Device::CUDA; t.device_index = 0;
  t.desc.dtype = DType::F32; t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape; t.desc.stride = make_rowmajor_strides(shape);
  return t;
}

} // namespace shim
} // namespace ai
