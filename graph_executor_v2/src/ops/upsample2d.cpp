#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "backends/cuda/ops/upsample2d/api.hpp"

namespace ai { namespace ops {

static inline bool is_nchw_f32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

int upsample2d_nearest_run(const Tensor& X, Tensor& Y,
                           const ai::Upsample2DAttrs& attrs,
                           StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return -2;
  // N,C 동일성 체크
  if (X.desc.shape[0]!=Y.desc.shape[0] || X.desc.shape[1]!=Y.desc.shape[1]) return -3;

  auto st = ai::Upsample2DNearestCudaLaunch(X, Y, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int upsample2d_nearest_backward_run(const Tensor& dY, Tensor& dX,
                                    const ai::Upsample2DAttrs& attrs,
                                    StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX)) return -2;
  // N,C 동일성 체크
  if (dY.desc.shape[0]!=dX.desc.shape[0] || dY.desc.shape[1]!=dX.desc.shape[1]) return -3;

  auto st = ai::Upsample2DNearestBackwardCudaLaunch(dY, dX, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
