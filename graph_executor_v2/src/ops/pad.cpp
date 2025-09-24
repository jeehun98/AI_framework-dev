#include "backends/cuda/ops/pad/api.hpp"

namespace ai { namespace ops {

static inline bool is_f32_cuda_rowmajor(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}

int pad_run(const Tensor& X, Tensor& Y, const ai::PadSpec& spec, StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(X) || !is_f32_cuda_rowmajor(Y)) return -2;
  auto st = ai::PadCudaLaunch(X, Y, spec, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int pad_backward_run(const Tensor& dY, Tensor& dX, const ai::PadSpec& spec, StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(dY) || !is_f32_cuda_rowmajor(dX)) return -2;
  auto st = ai::PadBackwardCudaLaunch(dY, dX, spec, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
