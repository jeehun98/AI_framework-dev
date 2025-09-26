#include <cuda_runtime.h>
#include "backends/cuda/ops/dropout/api.hpp"

namespace ai {

static inline bool is_rowmajor_2d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline bool is_rowmajor_2d_i32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// kernel launchers (통일된 시그니처/이름)
void dropout_forward_kernel_launcher(const float* X, float* Y, int32_t* mask,
                                     int M_rows, int N_cols, float p, bool scale_in_train,
                                     uint64_t seed, uint64_t counter_base, cudaStream_t s);
void dropout_backward_kernel_launcher(const float* dY, const int32_t* mask, float* dX,
                                      int M_rows, int N_cols, float p, bool scale_in_train,
                                      cudaStream_t s);

Status DropoutCudaLaunch(const Tensor& X, Tensor& Y, Tensor* mask,
                         const DropoutAttrs& attrs, StreamHandle stream)
{
  if (!is_rowmajor_2d_f32_cuda(X) || !is_rowmajor_2d_f32_cuda(Y))
    return Status::Invalid;
  if (X.desc.shape != Y.desc.shape) return Status::ShapeMismatch;

  const int M_rows = static_cast<int>(X.desc.shape[0]);
  const int N_cols = static_cast<int>(X.desc.shape[1]);

  int32_t* mask_ptr = nullptr;
  if (mask){
    if (!is_rowmajor_2d_i32_cuda(*mask)) return Status::Invalid;
    if (mask->desc.shape != X.desc.shape) return Status::ShapeMismatch;
    mask_ptr = static_cast<int32_t*>(mask->data);
  }

  dropout_forward_kernel_launcher(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    mask_ptr,
    M_rows, N_cols,
    attrs.p, attrs.scale_in_train, attrs.seed, attrs.counter_base,
    to_cuda(stream)
  );
  if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

Status DropoutCudaBackwardLaunch(const Tensor& dY, const Tensor& mask, Tensor& dX,
                                 const DropoutAttrs& attrs, StreamHandle stream)
{
  if (!is_rowmajor_2d_f32_cuda(dY) || !is_rowmajor_2d_f32_cuda(dX) || !is_rowmajor_2d_i32_cuda(mask))
    return Status::Invalid;
  if (dY.desc.shape != dX.desc.shape || dY.desc.shape != mask.desc.shape)
    return Status::ShapeMismatch;

  const int M_rows = static_cast<int>(dY.desc.shape[0]);
  const int N_cols = static_cast<int>(dY.desc.shape[1]);

  dropout_backward_kernel_launcher(
    static_cast<const float*>(dY.data),
    static_cast<const int32_t*>(mask.data),
    static_cast<float*>(dX.data),
    M_rows, N_cols,
    attrs.p, attrs.scale_in_train,
    to_cuda(stream)
  );
  if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

} // namespace ai
