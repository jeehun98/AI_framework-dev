// src/ops/layernorm.cpp
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "backends/cuda/ops/layernorm/api.hpp"  // ai::LayerNormAttrs, *_Launch

namespace ai { namespace ops {

static inline bool is_row_major_2d_f32(const Tensor& T) {
  return T.desc.dtype==DType::F32
      && T.desc.layout==Layout::RowMajor
      && T.desc.shape.size()==2;
}

// ── Forward: 검증 후 CUDA 런처 위임 (RMSNorm와 동일 패턴) ──
int layernorm_run(const Tensor& X,
                  const Tensor* gamma,
                  const Tensor* beta,
                  Tensor& Y,
                  const ai::LayerNormAttrs& attrs,
                  StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y)) return -2;
  if (X.device!=Device::CUDA || Y.device!=Device::CUDA)   return -3;
  if (X.desc.shape!=Y.desc.shape)                         return -4;

  const auto N = X.desc.shape[1];
  if (gamma && gamma->data &&
      !(gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N)) return -5;
  if (beta  && beta->data  &&
      !(beta->desc.shape.size()==1  && beta->desc.shape[0]==N))  return -6;

  auto st = LayerNormCudaLaunch(X, gamma, beta, Y, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7;
}

// ── Backward: 검증 후 CUDA 런처 위임 (RMSNorm와 동일 패턴) ──
int layernorm_backward_run(const Tensor& X,
                           const Tensor* gamma,
                           const Tensor& dY,
                           Tensor& dX,
                           Tensor* dgamma,
                           Tensor* dbeta,
                           const ai::LayerNormAttrs& attrs,
                           StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX)) return -2;
  if (X.desc.shape!=dY.desc.shape || X.desc.shape!=dX.desc.shape) return -3;

  const auto N = X.desc.shape[1];
  if (gamma && gamma->data &&
      !(gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N)) return -4;
  if (dgamma &&
      !(dgamma->desc.shape.size()==1 && dgamma->desc.shape[0]==N)) return -5;
  if (dbeta  &&
      !(dbeta->desc.shape.size()==1  && dbeta->desc.shape[0]==N))  return -6;

  auto st = LayerNormCudaBackwardLaunch(X, gamma, dY, dX, dgamma, dbeta, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
