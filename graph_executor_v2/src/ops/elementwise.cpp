#include "backends/cuda/ops/elementwise/api.hpp"

namespace ai { namespace ops {

static inline bool is_f32_cuda_rowmajor(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}

int ewise_unary_run(const Tensor& X, Tensor& Y,
                    UnaryOp op, const EWiseUnaryAttrs& attrs, StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(X) || !is_f32_cuda_rowmajor(Y)) return -2;
  auto st = ai::EWiseUnaryCudaLaunch(X, Y, op, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

int ewise_binary_run(const Tensor& A, const Tensor& B, Tensor& Y,
                     BinaryOp op, const EWiseBinaryAttrs& attrs, StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(A) || !is_f32_cuda_rowmajor(B) || !is_f32_cuda_rowmajor(Y)) return -2;
  auto st = ai::EWiseBinaryCudaLaunch(A, B, Y, op, attrs, stream);
  return (st==ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
