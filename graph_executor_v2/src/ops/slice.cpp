#include "backends/cuda/ops/slice/api.hpp"

namespace ai { namespace ops {

static inline bool is_f32_row_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}

int slice_run(const ai::Tensor& X, ai::Tensor& Y,
              const ai::SliceAttrs& attrs, StreamHandle s)
{
  if (!is_f32_row_cuda(X) || !is_f32_row_cuda(Y)) return -2;
  auto st = ai::SliceCudaLaunch(X, Y, attrs, s);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
