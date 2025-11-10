// backends/cuda/ops/batchnorm/launcher.cu
#include <cuda_runtime.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/batchnorm/api.hpp"
#include "backends/cuda/ops/batchnorm/bn_validate.hpp"

namespace ai {

// ===== extern kernel launchers (정의는 kernels.cu에서) =====

// Forward: per-channel batch mean/var (moments 기반)
void welford_reduce_meanvar_launcher(
    const float* X,                /* [N,C,H,W] or [N,H,W,C] */
    float* mean,                   /* [C] out */
    float* var,                    /* [C] out (batch var) */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Forward: normalize + optional affine(γ,β)
void bn_forward_normalize_affine_launcher(
    const float* X,
    const float* mean, const float* invstd,   /* [C] */
    const float* gamma, const float* beta,    /* [C] or nullptr if !affine */
    float* Y,                                 /* [N,C,H,W] or [N,H,W,C] */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Backward: dgamma/dbeta reductions
void bn_backward_reduce_dbeta_dgamma_launcher(
    const float* dY,           /* [N,C,H,W] or [N,H,W,C] */
    const float* X,            /* same layout */
    const float* mean,         /* [C] */
    const float* invstd,       /* [C] */
    float* dbeta, float* dgamma, /* [C] (둘 중 nullptr 허용) */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Backward: dX
void bn_backward_dx_launcher(
    const float* dY,           /* [N,C,H,W] or [N,H,W,C] */
    const float* X,            /* same layout */
    const float* mean,         /* [C] */
    const float* invstd,       /* [C] */
    const float* gamma,        /* [C] or nullptr if !affine */
    float* dX,                 /* [N,C,H,W] or [N,H,W,C] */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// ===== 작은 유틸: invstd = 1/sqrt(var + eps) =====
__global__ void compute_invstd_kernel(const float* __restrict__ var,
                                      float* __restrict__ invstd,
                                      float eps, int C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C) {
    invstd[i] = rsqrtf(var[i] + eps);
  }
}

// ===== running_* EMA 업데이트 커널 =====
// running = (1-m)*running + m*batch
__global__ void bn_update_running_kernel(float* __restrict__ running_mean,
                                         float* __restrict__ running_var,
                                         const float* __restrict__ batch_mean,
                                         const float* __restrict__ batch_var,
                                         float momentum,
                                         int C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C) {
    running_mean[i] = (1.f - momentum) * running_mean[i] + momentum * batch_mean[i];
    running_var[i]  = (1.f - momentum) * running_var[i]  + momentum * batch_var[i];
  }
}

// ===== Forward Launcher =====
Status BatchNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,
                           const Tensor* beta,
                           Tensor* running_mean,
                           Tensor* running_var,
                           Tensor& Y,
                           const BatchNormAttrs& a,
                           StreamHandle stream,
                           Tensor* save_mean,
                           Tensor* save_invstd,
                           const BatchNormWorkspaceFwd* ws_fwd)
{
  BNParsed d{};
  if (a.training) {
    AI_RETURN_IF_ERROR(BNValidateForwardTraining(
        X, Y, gamma, beta, running_mean, running_var, save_mean, save_invstd, a, d));
  } else {
    AI_RETURN_IF_ERROR(BNValidateForwardInference(
        X, Y, gamma, beta, running_mean, running_var, save_invstd, a, d));
  }

  // 현재 커널은 FP32 전용. (혼합정밀 커널 도입 시 완화)
  if (X.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return Status::DtypeMismatch;

  auto* s = as_cuda_stream(stream);
  const int N = d.N, C = d.C, H = d.H, W = d.W;
  const bool cl = a.channels_last;

  if (a.training) {
    // temp batch_var 버퍼: ws_fwd->blockbuf를 길이 >= C 로 요구
    if (!ws_fwd || !ws_fwd->blockbuf || (int)ws_fwd->blockbuf_elems < C)
      return Status::MissingInput;
    float* batch_var = ws_fwd->blockbuf; // [C]

    // 1) 배치 mean/var
    welford_reduce_meanvar_launcher(
        static_cast<const float*>(X.data),
        static_cast<float*>(save_mean->data),
        batch_var,
        N, C, H, W, cl, s);

    // 2) invstd = 1/sqrt(var + eps)
    {
      dim3 block(256);
      dim3 grid((C + block.x - 1) / block.x);
      AI_CUDA_CHECK_LAUNCH((
        compute_invstd_kernel<<<grid, block, 0, s>>>(
          batch_var,
          static_cast<float*>(save_invstd->data),
          a.eps, C)));
    }

    // 3) Y = (X - mean) * invstd * gamma + beta
    bn_forward_normalize_affine_launcher(
        static_cast<const float*>(X.data),
        static_cast<const float*>(save_mean->data),
        static_cast<const float*>(save_invstd->data),
        a.with_affine ? static_cast<const float*>(gamma->data) : nullptr,
        a.with_affine ? static_cast<const float*>(beta->data)  : nullptr,
        static_cast<float*>(Y.data),
        N, C, H, W, cl, s);

    // 4) running_* EMA = (1-m)*run + m*batch
    {
      dim3 block(256);
      dim3 grid((C + block.x - 1) / block.x);
      AI_CUDA_CHECK_LAUNCH((
        bn_update_running_kernel<<<grid, block, 0, s>>>(
            static_cast<float*>(running_mean->data),
            static_cast<float*>(running_var->data),
            static_cast<const float*>(save_mean->data),
            batch_var,                // 주의: invstd가 아니라 배치 분산
            a.momentum, C)));
    }
  } else {
    // Inference: running_* 로 invstd 만들고 정규화
    {
      dim3 block(256);
      dim3 grid((C + block.x - 1) / block.x);
      AI_CUDA_CHECK_LAUNCH((
        compute_invstd_kernel<<<grid, block, 0, s>>>(
          static_cast<const float*>(running_var->data),
          static_cast<float*>(save_invstd->data),  // 캡처-세이프 목적지
          a.eps, C)));
    }

    bn_forward_normalize_affine_launcher(
        static_cast<const float*>(X.data),
        static_cast<const float*>(running_mean->data),
        static_cast<const float*>(save_invstd->data),
        a.with_affine ? static_cast<const float*>(gamma->data) : nullptr,
        a.with_affine ? static_cast<const float*>(beta->data)  : nullptr,
        static_cast<float*>(Y.data),
        N, C, H, W, cl, s);
  }

  return Status::Ok;
}

// ===== Backward Launcher =====
Status BatchNormCudaBackwardLaunch(const Tensor& dY,
                                   const Tensor& X,
                                   const Tensor* gamma,
                                   const Tensor& save_mean,
                                   const Tensor& save_invstd,
                                   Tensor* dX,
                                   Tensor* dgamma,
                                   Tensor* dbeta,
                                   const BatchNormAttrs& a,
                                   StreamHandle stream,
                                   const BatchNormWorkspaceBwd* /*ws_bwd*/)
{
  BNParsed d{};
  AI_RETURN_IF_ERROR(BNValidateBackward(
      dY, X, gamma, save_mean, save_invstd, dX, dgamma, dbeta, a, d));

  // 커널은 FP32 전용
  if (dY.desc.dtype != DType::F32 || X.desc.dtype != DType::F32) return Status::DtypeMismatch;
  if (dX && dX->desc.dtype != DType::F32) return Status::DtypeMismatch;

  auto* s = as_cuda_stream(stream);

  // (옵션) dgamma/dbeta 누적 버퍼 0 초기화 책임이 호출자에게 있다는 계약을 유지
  // 필요 시 다음을 활성화:
  // if (dgamma) AI_CUDA_CHECK(cudaMemsetAsync(dgamma->data, 0, d.C * sizeof(float), s));
  // if (dbeta ) AI_CUDA_CHECK(cudaMemsetAsync(dbeta->data,  0, d.C * sizeof(float), s));

  // 1) dgamma, dbeta 감소
  if (dgamma || dbeta) {
    bn_backward_reduce_dbeta_dgamma_launcher(
        static_cast<const float*>(dY.data),
        static_cast<const float*>(X.data),
        static_cast<const float*>(save_mean.data),
        static_cast<const float*>(save_invstd.data),
        dbeta  ? static_cast<float*>(dbeta->data)  : nullptr,
        dgamma ? static_cast<float*>(dgamma->data) : nullptr,
        d.N, d.C, d.H, d.W, a.channels_last, s);
  }

  // 2) dX
  if (dX) {
    bn_backward_dx_launcher(
        static_cast<const float*>(dY.data),
        static_cast<const float*>(X.data),
        static_cast<const float*>(save_mean.data),
        static_cast<const float*>(save_invstd.data),
        (a.with_affine && gamma) ? static_cast<const float*>(gamma->data) : nullptr,
        static_cast<float*>(dX->data),
        d.N, d.C, d.H, d.W, a.channels_last, s);
  }

  return Status::Ok;
}

} // namespace ai
