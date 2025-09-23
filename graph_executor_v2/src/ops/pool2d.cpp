// src/ops/pool2d.cpp
#include "backends/cuda/ops/pool2d/api.hpp"

namespace ai { namespace ops {

static inline bool is_nchw_f32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is_nchw_i32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

int maxpool2d_run(const Tensor& X, Tensor& Y, Tensor* Indices,
                  const ai::Pool2DAttrs& attrs, StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return -2;
  if (Indices && !is_nchw_i32_4d_cuda(*Indices)) return -3;
  auto st = ai::MaxPool2DCudaLaunch(X, Y, Indices, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int maxpool2d_backward_run(const Tensor& dY, const Tensor& Indices, Tensor& dX,
                           const ai::Pool2DAttrs& attrs, StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX) || !is_nchw_i32_4d_cuda(Indices)) return -2;
  auto st = ai::MaxPool2DBackwardCudaLaunch(dY, Indices, dX, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int avgpool2d_run(const Tensor& X, Tensor& Y,
                  const ai::Pool2DAttrs& attrs, StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return -2;
  auto st = ai::AvgPool2DCudaLaunch(X, Y, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int avgpool2d_backward_run(const Tensor& dY, Tensor& dX,
                           const ai::Pool2DAttrs& attrs, StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX)) return -2;
  auto st = ai::AvgPool2DBackwardCudaLaunch(dY, dX, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
