#include "backends/cuda/ops/view/api.hpp"

namespace ai { namespace ops {

// 이 레벨에서는 단순하게 API를 그대로 노출(디스패치가 여러 백엔드를 다룰 예정이면
// device 분기만 추가). 현 구조에선 CUDA만 있으니 바로 호출.

int permute_make_view(const Tensor& X, const std::vector<int>& perm, Tensor& Yout) {
  TensorDesc outd{};
  auto st = ai::PermuteMakeView(X.desc, perm, outd);
  if (st != ai::Status::Ok) return -2;
  Yout = ai::MakeViewTensor(X.data, outd, X.device, X.device_index);
  return 0;
}

int transpose2d_make_view(const Tensor& X, int d0, int d1, Tensor& Yout) {
  TensorDesc outd{};
  auto st = ai::Transpose2DMakeView(X.desc, d0, d1, outd);
  if (st != ai::Status::Ok) return -2;
  Yout = ai::MakeViewTensor(X.data, outd, X.device, X.device_index);
  return 0;
}

int expand_make_view(const Tensor& X, const std::vector<int64_t>& out_shape, Tensor& Yout) {
  TensorDesc outd{};
  auto st = ai::ExpandMakeView(X.desc, out_shape, outd);
  if (st != ai::Status::Ok) return -2;
  Yout = ai::MakeViewTensor(X.data, outd, X.device, X.device_index);
  return 0;
}

}} // namespace ai::ops
