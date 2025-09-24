#include "backends/cuda/ops/concat/api.hpp"

namespace ai { namespace ops {

static inline bool is_f32_row_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}

int concat_run(const std::vector<ai::Tensor>& Xs,
               ai::Tensor& Y,
               const ai::ConcatAttrs& attrs,
               StreamHandle s)
{
  if (Xs.empty()) return -1;
  for (auto& X : Xs) if (!is_f32_row_cuda(X)) return -2;
  if (!is_f32_row_cuda(Y)) return -2;

  auto st = ai::ConcatCudaLaunch(Xs, Y, attrs, s);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
