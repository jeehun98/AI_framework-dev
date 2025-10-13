#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <type_traits>

namespace ai {

enum class Device { CPU, CUDA };
enum class DType  { F32, F16, BF16, I32, I8 };
enum class Layout { RowMajor, ColMajor };

// ---- dtype size helper ----
inline std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32:  return 4;
    case DType::F16:  return 2;
    case DType::BF16: return 2;
    case DType::I32:  return 4;
    case DType::I8:   return 1;
    default:          return 0;
  }
}

// ---- row-major stride builder ----
inline std::vector<int64_t> make_rowmajor_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size());
  int64_t st = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    s[static_cast<size_t>(i)] = st;
    st *= shape[static_cast<size_t>(i)];
  }
  return s;
}

struct TensorDesc {
  DType  dtype  { DType::F32 };
  Layout layout { Layout::RowMajor };
  std::vector<int64_t> shape;   // e.g., [M,K]
  std::vector<int64_t> stride;  // elems, not bytes

  int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
};

struct Tensor {
  void*   data{nullptr};
  TensorDesc desc{};
  Device  device{Device::CUDA};
  int     device_index{0}; // GPU id if CUDA

  // ---- predicates ----
  bool is_defined() const { return data != nullptr; }
  bool is_cuda()    const { return device == Device::CUDA; }

  bool is_contiguous_rowmajor_2d() const {
    if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
    if (desc.stride.size() != 2) return false;
    const int64_t rows = desc.shape[0];
    const int64_t cols = desc.shape[1];
    return (desc.stride[1] == 1) && (desc.stride[0] == cols) && (rows >= 0 && cols >= 0);
  }

  // ---- sizes ----
  int64_t numel() const {
    if (desc.shape.empty()) return 0;
    int64_t n = 1;
    for (auto v : desc.shape) {
      if (v < 0) return 0;
      n *= v;
    }
    return n;
  }

  int64_t nbytes() const {
    return static_cast<int64_t>(numel()) * static_cast<int64_t>(dtype_size(desc.dtype));
    }

  // ---- raw data ptr ----
  void*       data_ptr()       { return data; }
  const void* data_ptr() const { return data; }

  // ---- typed data ptr (non-const) ----
  template <typename T>
  T* data_ptr() {
    static_assert(!std::is_const<T>::value, "Use const overload for const type");
    return reinterpret_cast<T*>(data);
  }

  // ---- typed data ptr (const) ----
  template <typename T>
  const T* data_ptr() const {
    return reinterpret_cast<const T*>(data);
  }
};

// ---- GEMM helpers (2D leading dims) ----
inline int64_t lda(const Tensor& A){ // [M,K]
  return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
inline int64_t ldb(const Tensor& B){ // [K,N]
  return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
inline int64_t ldd(const Tensor& D){ // [M,N]
  return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

// ---- convenient makers (optional) ----
inline Tensor make_tensor2d(void* ptr, int64_t rows, int64_t cols) {
  Tensor t;
  t.data         = ptr;
  t.device       = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype   = DType::F32;
  t.desc.layout  = Layout::RowMajor;
  t.desc.shape   = {rows, cols};
  t.desc.stride  = {cols, 1};
  return t;
}

inline Tensor make_tensor_from_ptr(void* ptr, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data         = ptr;
  t.device       = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype   = DType::F32;
  t.desc.layout  = Layout::RowMajor;
  t.desc.shape   = shape;
  t.desc.stride  = make_rowmajor_strides(shape);
  return t;
}

} // namespace ai
