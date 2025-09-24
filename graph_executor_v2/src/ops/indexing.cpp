#include "backends/cuda/ops/indexing/api.hpp"

namespace ai { namespace ops {

int gather_run(const Tensor& X, const Tensor& Index, Tensor& Y, int axis, StreamHandle s) {
  auto st = ai::GatherCudaLaunch(X, Index, Y, axis, s);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int scatter_add_run(Tensor& Out, const Tensor& Index, const Tensor& Src, int axis, StreamHandle s) {
  auto st = ai::ScatterAddCudaLaunch(Out, Index, Src, axis, s);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
