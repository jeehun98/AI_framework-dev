#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

namespace ai {

enum class Device { CPU, CUDA };
enum class DType  { F32, F16, BF16, I32, I8 };
enum class Layout { RowMajor, ColMajor };

struct TensorDesc {
  DType dtype;
  Layout layout;
  std::vector<int64_t> shape;   // e.g., [M,K]
  std::vector<int64_t> stride;  // elems, not bytes

  int64_t dim(int i) const { return shape.at(i); }
};

struct Tensor {
  void* data{nullptr};
  TensorDesc desc;
  Device device{Device::CPU};
  int device_index{0}; // GPU id if CUDA

  bool is_cuda() const { return device == Device::CUDA; }
  bool is_contiguous_rowmajor_2d() const {
    if (desc.shape.size()!=2 || desc.layout!=Layout::RowMajor) return false;
    return desc.stride.size()==2 && desc.stride[1]==1;
  }
};

// helpers (GEMM 2D)
inline int64_t lda(const Tensor& A){ // [M,K]
  return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
inline int64_t ldb(const Tensor& B){ // [K,N]
  return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
inline int64_t ldd(const Tensor& D){ // [M,N]
  return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

} // namespace ai
