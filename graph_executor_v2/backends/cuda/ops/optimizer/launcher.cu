#include <cuda_runtime.h>
#include "backends/cuda/ops/optimizer/api.hpp"

namespace ai {

// ---- helpers (RMSNorm launcher 스타일과 동일한 톤) ----
static inline bool is_1d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==1 &&
         t.data!=nullptr;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }
static inline int64_t numel_1d(const Tensor& t){ return (int64_t)t.desc.shape[0]; }

// ---- raw kernel launchers (kernels.cu 쪽에 구현) ----
void sgd_update_kernel_launcher(
    float* P,                      // [N]
    const float* G,                // [N]
    float* V,                      // [N] or null
    int64_t N,
    float lr, float momentum, float dampening,
    int nesterov, float weight_decay,
    cudaStream_t s);

void adamw_update_kernel_launcher(
    float* P, const float* G, float* M, float* V, // [N] each
    int64_t N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int bias_correction, int step,
    cudaStream_t s);

// ============================ SGD ============================
Status SGDCudaUpdateLaunch(Tensor& P,
                           const Tensor& G,
                           Tensor* V,
                           const SGDAttrs& attrs,
                           StreamHandle stream)
{
  // 타입/디바이스/shape 체크
  if (!is_1d_f32_cuda(P) || !is_1d_f32_cuda(G)) return Status::Invalid;
  if (numel_1d(P) != numel_1d(G)) return Status::ShapeMismatch;

  const bool use_momentum = (attrs.momentum > 0.0f);
  if (use_momentum) {
    if (!V || !is_1d_f32_cuda(*V)) return Status::Invalid;
    if (numel_1d(*V) != numel_1d(P)) return Status::ShapeMismatch;
  }

  // 파라미터 범위 간단 검증 (선택)
  if (attrs.lr <= 0.0f) return Status::Invalid;
  if (attrs.momentum < 0.0f || attrs.momentum >= 1.0f) return Status::Invalid;
  if (attrs.dampening < 0.0f || attrs.dampening >= 1.0f) return Status::Invalid;
  if (attrs.weight_decay < 0.0f) return Status::Invalid;

  sgd_update_kernel_launcher(
      static_cast<float*>(P.data),
      static_cast<const float*>(G.data),
      (use_momentum ? static_cast<float*>(V->data) : nullptr),
      numel_1d(P),
      attrs.lr, attrs.momentum, attrs.dampening,
      attrs.nesterov ? 1 : 0, attrs.weight_decay,
      to_cuda(stream)
  );

  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess)? Status::Ok : Status::RuntimeError;
}

// ============================ AdamW ============================
Status AdamWCudaUpdateLaunch(Tensor& P,
                             const Tensor& G,
                             Tensor& M,
                             Tensor& V,
                             const AdamWAttrs& attrs,
                             StreamHandle stream)
{
  if (!is_1d_f32_cuda(P) || !is_1d_f32_cuda(G) ||
      !is_1d_f32_cuda(M) || !is_1d_f32_cuda(V))
    return Status::Invalid;

  const int64_t N = numel_1d(P);
  if (numel_1d(G)!=N || numel_1d(M)!=N || numel_1d(V)!=N)
    return Status::ShapeMismatch;

  // 파라미터 검증
  if (attrs.lr <= 0.0f) return Status::Invalid;
  if (attrs.beta1 < 0.0f || attrs.beta1 >= 1.0f) return Status::Invalid;
  if (attrs.beta2 < 0.0f || attrs.beta2 >= 1.0f) return Status::Invalid;
  if (attrs.eps <= 0.0f) return Status::Invalid;
  if (attrs.weight_decay < 0.0f) return Status::Invalid;
  if (attrs.bias_correction && attrs.step < 1) return Status::Invalid;

  adamw_update_kernel_launcher(
      static_cast<float*>(P.data),
      static_cast<const float*>(G.data),
      static_cast<float*>(M.data),
      static_cast<float*>(V.data),
      N,
      attrs.lr, attrs.beta1, attrs.beta2, attrs.eps,
      attrs.weight_decay,
      attrs.bias_correction ? 1 : 0,
      attrs.step,
      to_cuda(stream)
  );

  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess)? Status::Ok : Status::RuntimeError;
}

} // namespace ai
