#include "backends/cuda/ops/reduction/api.hpp"

namespace ai { namespace ops {

static inline bool is_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}

int reduce_run(const Tensor& X, Tensor& Y, const ai::ReduceAttrs& attrs, StreamHandle stream)
{
  if (!is_f32_cuda(X) || !is_f32_cuda(Y)) return -2;
  auto st = ai::ReduceCudaLaunch(X, Y, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
