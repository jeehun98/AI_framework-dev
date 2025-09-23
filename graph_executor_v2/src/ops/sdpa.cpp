#include "backends/cuda/ops/sdpa/api.hpp"

namespace ai { namespace ops {

static inline bool is4d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

int sdpa_run(const Tensor& Q, const Tensor& K, const Tensor& V,
             const Tensor* mask, Tensor& Y,
             const ai::SDPAAttrs& attrs, StreamHandle stream)
{
  if (!is4d_f32_cuda(Q) || !is4d_f32_cuda(K) || !is4d_f32_cuda(V) || !is4d_f32_cuda(Y)) return -2;
  auto st = ai::SDPACudaLaunch(Q,K,V,mask,Y,attrs,stream);
  return (st==ai::Status::Ok) ? 0 : (st==ai::Status::Unimplemented ? -8 : -7);
}

int sdpa_backward_run(const Tensor& Q, const Tensor& K, const Tensor& V,
                      const Tensor* mask, const Tensor& dY,
                      Tensor* dQ, Tensor* dK, Tensor* dV,
                      const ai::SDPAAttrs& attrs, StreamHandle stream)
{
  auto st = ai::SDPACudaBackwardLaunch(Q,K,V,mask,dY,dQ,dK,dV,attrs,stream);
  return (st==ai::Status::Ok) ? 0 : (st==ai::Status::Unimplemented ? -8 : -7);
}

}} // namespace ai::ops
