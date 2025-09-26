#include "backends/cuda/ops/sdpa/api.hpp"

// 필요 시 다른 ops API 포함 (gemm/softmax 등은 launcher에서 직접 포함)
namespace ai { namespace ops {

static inline bool is4d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

// BHxD 규약을 사용한다면 별도 검증 훅
static inline bool is_bhxd_f32_4d_cuda(const Tensor& t){
  return is4d_f32_cuda(t);
}

// ------------ Forward Wrapper ------------
int sdpa_run(const Tensor& Q, const Tensor& K, const Tensor& V,
             const Tensor* mask, Tensor& Y,
             const ai::SDPAAttrs& attrs, StreamHandle stream)
{
  if (!is4d_f32_cuda(Q) || !is4d_f32_cuda(K) || !is4d_f32_cuda(V) || !is4d_f32_cuda(Y)) return -2;
  auto st = ai::SDPACudaLaunch(Q,K,V,mask,Y,attrs,stream);
  return (st==ai::Status::Ok) ? 0 : (st==ai::Status::Unimplemented ? -8 : -7);
}

// ------------ Backward Wrapper ------------
int sdpa_backward_run(const Tensor& Q, const Tensor& K, const Tensor& V,
                      const Tensor& dY,
                      Tensor* dQ, Tensor* dK, Tensor* dV,
                      const ai::SDPAAttrs& attrs,
                      StreamHandle stream,
                      const Tensor* mask = nullptr)
{
  if (!is_bhxd_f32_4d_cuda(Q) || !is_bhxd_f32_4d_cuda(K) || !is_bhxd_f32_4d_cuda(V) ||
      !is_bhxd_f32_4d_cuda(dY)) return -2;

  // 필요한 그래디언트 포인터가 하나도 없으면 에러 (정책에 따라 조정 가능)
  if (!dQ && !dK && !dV) return -2;

  
  auto st = ai::SDPACudaBackwardLaunch(Q, K, V, dY, mask, dQ, dK, dV, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7;
}

}} // namespace ai::ops
