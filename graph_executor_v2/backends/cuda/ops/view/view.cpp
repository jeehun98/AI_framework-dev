// backends/cuda/ops/view/view.cpp
#include "backends/cuda/ops/view/api.hpp"
namespace ai {
static inline bool is_f32_row(const Tensor&t){return t.device==Device::CUDA&&t.desc.dtype==DType::F32&&t.desc.layout==Layout::RowMajor;}
Status ViewAliasCheck(const Tensor& X, const Tensor& Y, const ViewAttrs& a){
  if (!is_f32_row(X) || !is_f32_row(Y)) return Status::Invalid;
  if ((int)Y.desc.shape.size()!=a.rank) return Status::ShapeMismatch;
  int64_t expect=1, got=1;
  for (auto v:X.desc.shape) expect*=v;
  for (int i=0;i<a.rank;++i) got*=a.shape[i];
  if (expect!=got) return Status::ShapeMismatch;
  // 별도 제약: alias를 원하면 포인터 동일성 확인(선택)
  if (X.data!=Y.data) return Status::Invalid; // alias만 허용
  return Status::Ok;
}
} // namespace ai
