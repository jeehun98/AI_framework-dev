#include "backends/cuda/ops/conv2d/api.hpp"

namespace ai { namespace ops {

static inline bool is4_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is1_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}

int conv2d_run(const Tensor& X, const Tensor& W, const Tensor* B, Tensor& Y,
               const ai::Conv2DAttrs& attrs, StreamHandle stream)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(W) || !is4_f32_cuda(Y)) return -2;
  auto st = ai::Conv2DCudaLaunch(X, W, B, Y, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int conv2d_backward_run(const Tensor& X, const Tensor& W, const Tensor& dY,
                        Tensor* dW, Tensor* dB, Tensor* dX,
                        const ai::Conv2DAttrs& attrs, StreamHandle stream)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(W) || !is4_f32_cuda(dY)) return -2;
  if (dW && !is4_f32_cuda(*dW)) return -2;
  if (dB && !is1_f32_cuda(*dB)) return -2;
  if (dX && !is4_f32_cuda(*dX)) return -2;

  auto st = ai::Conv2DCudaBackwardLaunch(X, W, dY, dW, dB, dX, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
