// backends/cuda/ops/dropout/launcher.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "backends/cuda/ops/dropout/api.hpp"

namespace ai {

static inline bool is_row_major_f32_2d_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline bool is_row_major_i32_2d_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ---- kernels decl ----
void dropout_fwd_kernel_launcher(const float* x, float* y, int32_t* mask,
                                 int M, int N,
                                 float p, bool scale_in_train,
                                 uint64_t seed, uint64_t counter_base,
                                 cudaStream_t s);

void dropout_bwd_kernel_launcher(const float* dy, const int32_t* mask, float* dx,
                                 int M, int N,
                                 float p, bool scale_in_train,
                                 cudaStream_t s);

// ---- forward ----
Status DropoutCudaLaunch(const Tensor& X, Tensor& Y, Tensor* Mask,
                         const DropoutAttrs& attrs, StreamHandle stream)
{
  if (!is_row_major_f32_2d_cuda(X) || !is_row_major_f32_2d_cuda(Y))
    return Status::Invalid;
  if (X.desc.shape != Y.desc.shape) return Status::ShapeMismatch;

  int M = (int)X.desc.shape[0];
  int N = (int)X.desc.shape[1];

  int32_t* mask_ptr = nullptr;
  if (Mask){
    if (!is_row_major_i32_2d_cuda(*Mask)) return Status::Invalid;
    if (Mask->desc.shape != Y.desc.shape) return Status::ShapeMismatch;
    mask_ptr = static_cast<int32_t*>(Mask->data);
  }

  dropout_fwd_kernel_launcher(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      mask_ptr,
      M, N,
      attrs.p, attrs.scale_in_train,
      attrs.seed, attrs.counter_base,
      to_cuda(stream)
  );
  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// ---- backward ----
Status DropoutCudaBackwardLaunch(const Tensor& dY, const Tensor& Mask, Tensor& dX,
                                 const DropoutAttrs& attrs, StreamHandle stream)
{
  (void)attrs; // bwd는 RNG 불필요(마스크만 사용)
  if (!is_row_major_f32_2d_cuda(dY) ||
      !is_row_major_f32_2d_cuda(dX) ||
      !is_row_major_i32_2d_cuda(Mask))
    return Status::Invalid;

  if (dY.desc.shape != dX.desc.shape ||
      dY.desc.shape != Mask.desc.shape) return Status::ShapeMismatch;

  int M = (int)dY.desc.shape[0];
  int N = (int)dY.desc.shape[1];

  dropout_bwd_kernel_launcher(
      static_cast<const float*>(dY.data),
      static_cast<const int32_t*>(Mask.data),
      static_cast<float*>(dX.data),
      M, N,
      /*p*/0.0f, /*scale_in_train unused here*/ true,
      to_cuda(stream)
  );
  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
