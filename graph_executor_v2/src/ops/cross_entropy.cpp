// src/ops/cross_entropy.cpp
#include "backends/cuda/ops/cross_entropy/api.hpp"

namespace ai { namespace ops {

static inline bool is_rowmajor_2d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}

static inline bool is_rowmajor_1d_i32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==1;
}

static inline bool is_rowmajor_1d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==1;
}

// Forward: X:[M,N] logits, target:[M] (I32), loss:[M] or [1]
int cross_entropy_run(const Tensor& X,
                      const Tensor& target,
                      Tensor&       loss,
                      const ai::CrossEntropyAttrs& attrs,
                      StreamHandle  stream)
{
  if (!is_rowmajor_2d_f32_cuda(X) || !is_rowmajor_1d_i32_cuda(target))
    return -2;

  const int64_t M = X.desc.shape[0];

  // ✅ target 길이 검증 추가
  if (target.desc.shape[0] != M)
    return -3;

  const bool want_vec = (attrs.reduction == ai::Reduction::None);
  if (want_vec) {
    if (!is_rowmajor_1d_f32_cuda(loss)) return -2;
    if (loss.desc.shape[0] != M) return -3;
  } else {
    if (!is_rowmajor_1d_f32_cuda(loss)) return -2;
    if (loss.desc.shape[0] != 1) return -3;
  }

  auto st = ai::CrossEntropyCudaLaunch(X, target, loss, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

// Backward: dX shape == X
int cross_entropy_backward_run(const Tensor& X,
                               const Tensor& target,
                               Tensor&       dX,
                               const ai::CrossEntropyAttrs& attrs,
                               StreamHandle  stream)
{
  if (!is_rowmajor_2d_f32_cuda(X) || !is_rowmajor_2d_f32_cuda(dX) || !is_rowmajor_1d_i32_cuda(target))
    return -2;

  // ✅ target 길이 검증 추가
  if (target.desc.shape[0] != X.desc.shape[0])
    return -3;

  if (X.desc.shape != dX.desc.shape)
    return -3;

  auto st = ai::CrossEntropyCudaBackwardLaunch(X, target, dX, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
