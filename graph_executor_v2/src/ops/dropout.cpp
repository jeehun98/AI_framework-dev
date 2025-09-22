// src/ops/dropout.cpp
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "backends/cuda/ops/dropout/api.hpp"

namespace ai { namespace ops {

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

// Forward: X -> Y, (선택) mask
int dropout_run(const Tensor& X,
                Tensor& Y,
                Tensor* mask,  // nullable
                const ai::DropoutAttrs& attrs,
                StreamHandle stream)
{
  // 형식/장치 검증
  if (!is_rowmajor_2d_f32_cuda(X) || !is_rowmajor_2d_f32_cuda(Y)) return -2;
  if (X.desc.shape != Y.desc.shape) return -3;

  if (mask) {
    if (!is_rowmajor_2d_i32_cuda(*mask)) return -2;
    if (mask->desc.shape != X.desc.shape) return -3;
  }

  ai::Status st = ai::DropoutCudaLaunch(X, Y, mask, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7;
}

// Backward: dY, mask -> dX
int dropout_backward_run(const Tensor& dY,
                         const Tensor& mask,
                         Tensor& dX,
                         const ai::DropoutAttrs& attrs,
                         StreamHandle stream)
{
  if (!is_rowmajor_2d_f32_cuda(dY) || !is_rowmajor_2d_f32_cuda(dX) || !is_rowmajor_2d_i32_cuda(mask)) return -2;
  if (dY.desc.shape != dX.desc.shape || dY.desc.shape != mask.desc.shape) return -3;

  ai::Status st = ai::DropoutCudaBackwardLaunch(dY, mask, dX, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
