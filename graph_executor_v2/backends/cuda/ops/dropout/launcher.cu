#include <cuda_runtime.h>
#include <algorithm>
#include "backends/cuda/ops/dropout/api.hpp"

// kernels.cu 에서 노출한 C-링크 런처들
extern "C" void dropout_forward_kernel_launcher(
    const float* x, float* y, int32_t* mask,
    size_t n,
    float p, bool scale_in_train,
    uint64_t seed, uint64_t counter_base,
    cudaStream_t s);

extern "C" void dropout_backward_kernel_launcher(
    const float* gy, const int32_t* mask, float* gx,
    size_t n,
    float p, bool scale_in_train,
    cudaStream_t s);

namespace ai {

static inline bool is_f32_cuda_rowmajor(const Tensor& t) {
  return t.device == Device::CUDA &&
         t.desc.dtype == DType::F32 &&
         t.desc.layout == Layout::RowMajor;
}
static inline bool is_i32_cuda_rowmajor(const Tensor& t) {
  return t.device == Device::CUDA &&
         t.desc.dtype == DType::I32 &&
         t.desc.layout == Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h) {
  return reinterpret_cast<cudaStream_t>(h);
}
static inline size_t numel(const Tensor& t) {
  size_t n = 1;
  for (auto s : t.desc.shape) n *= (size_t)s;
  return n;
}

Status DropoutCudaLaunch(const Tensor& X,
                         Tensor& Y,
                         Tensor* mask,            // may be null; if provided -> I32
                         const DropoutAttrs& attrs,
                         StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(X) || !is_f32_cuda_rowmajor(Y)) return Status::Invalid;
  if (X.desc.shape != Y.desc.shape) return Status::ShapeMismatch;
  if (mask) {
    if (!is_i32_cuda_rowmajor(*mask)) return Status::Invalid;
    if (mask->desc.shape != X.desc.shape) return Status::ShapeMismatch;
  }
  if (!(attrs.p >= 0.f && attrs.p < 1.f)) return Status::Invalid;

  const size_t n = numel(X);
  auto s = to_cuda(stream);

  dropout_forward_kernel_launcher(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      mask ? static_cast<int32_t*>(mask->data) : nullptr,
      n,
      attrs.p,
      attrs.scale_in_train,
      attrs.seed,
      attrs.counter_base,
      s
  );

  // (선택) 런치 에러 체크
  // if (auto err = cudaGetLastError(); err != cudaSuccess) return Status::CudaError;

  return Status::Ok;
}

Status DropoutCudaBackwardLaunch(const Tensor& dY,
                                 const Tensor& mask,
                                 Tensor& dX,
                                 const DropoutAttrs& attrs,
                                 StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(dY) || !is_f32_cuda_rowmajor(dX)) return Status::Invalid;
  if (!is_i32_cuda_rowmajor(mask)) return Status::Invalid;
  if (dY.desc.shape != dX.desc.shape || dY.desc.shape != mask.desc.shape) return Status::ShapeMismatch;

  const size_t n = numel(dY);
  auto s = to_cuda(stream);

  dropout_backward_kernel_launcher(
      static_cast<const float*>(dY.data),
      static_cast<const int32_t*>(mask.data),
      static_cast<float*>(dX.data),
      n,
      attrs.p,
      attrs.scale_in_train,
      s
  );

  // (선택) 런치 에러 체크
  // if (auto err = cudaGetLastError(); err != cudaSuccess) return Status::CudaError;

  return Status::Ok;
}

} // namespace ai
