#include "backends/cuda/ops/softmax/api.hpp"

namespace ai { namespace ops {

static inline bool is_row_major_2d_f32(const Tensor& t){
  return t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor && t.desc.shape.size()==2;
}
int softmax_run(const Tensor& X, const Tensor* Mask, Tensor& Y,
                const ai::SoftmaxAttrs& attrs, StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y)) return -2;
  if (X.device!=Device::CUDA || Y.device!=Device::CUDA)   return -3;
  if (X.desc.shape!=Y.desc.shape)                         return -4;
  auto st = ai::SoftmaxCudaLaunch(X, Mask, Y, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

// ✅ 레거시 시그니처: 외부 TU는 이걸만 부르게 유지
int softmax_run(const Tensor& X, const Tensor* Mask, Tensor& Y,
                float scale, bool log, StreamHandle stream)
{
  ai::SoftmaxAttrs attrs{};
  attrs.scale = scale;
  attrs.log   = log;
  return softmax_run(X, Mask, Y, attrs, stream);
}

int softmax_backward_run(const Tensor& Y, const Tensor& dY, Tensor& dX,
                         const ai::SoftmaxAttrs& attrs, StreamHandle stream)
{
  if (!is_row_major_2d_f32(Y) || !is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX)) return -2;
  if (Y.device!=Device::CUDA || dY.device!=Device::CUDA || dX.device!=Device::CUDA)     return -3;
  if (Y.desc.shape!=dY.desc.shape || Y.desc.shape!=dX.desc.shape)                       return -4;
  auto st = ai::SoftmaxCudaBackwardLaunch(Y, /*Mask*/nullptr, dY, dX, attrs, /*y_provided*/true, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}


}} // namespace ai::ops
