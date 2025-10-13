// backends/cuda/ops/batchnorm/launcher.cu
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cassert>

#include "backends/cuda/ops/batchnorm/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif

namespace ai {

// ===== utils =====
static inline bool is4_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is1_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ===== extern kernel launchers (정의는 kernels.cu에서) =====
// Forward: mean/var 감소 (Welford or 2-pass)
void welford_reduce_meanvar_launcher(
    const float* X, /* [N,C,H,W] or [N,H,W,C] */
    float* mean,    /* [C] out */
    float* var,     /* [C] out (biased or unbiased는 구현에서 결정; 보통 unbiased로) */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Forward: 정규화 + (옵션) affine(γ,β) + (옵션) save_invstd
void bn_forward_normalize_affine_launcher(
    const float* X,
    const float* mean, const float* invstd,  /* [C] */
    const float* gamma, const float* beta,   /* [C] or nullptr if !affine */
    float* Y,                                 /* [N,C,H,W] or [N,H,W,C] */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Forward: running 통계 갱신 (training)
__global__ void bn_update_running_kernel(
    float* running_mean, float* running_var,
    const float* batch_mean, const float* batch_var,
    float momentum, int C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C) {
    float rm = running_mean[i];
    float rv = running_var[i];
    float bm = batch_mean[i];
    float bv = batch_var[i];
    running_mean[i] = (1.f - momentum)*rm + momentum*bm;
    running_var[i]  = (1.f - momentum)*rv + momentum*bv;
  }
}

// Backward: dgamma/dbeta 감소
void bn_backward_reduce_dbeta_dgamma_launcher(
    const float* dY,           /* [N,C,H,W] or [N,H,W,C] (여기서는 post-activation이 아닌, BN의 dY) */
    const float* X,            /* same layout */
    const float* mean,         /* [C] */
    const float* invstd,       /* [C] */
    float* dbeta, float* dgamma, /* [C] (둘 중 nullptr 허용) */
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s);

// Backward: dX 계산
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
                           const BatchNormWorkspaceFwd* /*ws_fwd*/)
{
  // ---- 검증 ----
  if (!is4_f32_cuda(X) || !is4_f32_cuda(Y)) return Status::Invalid;
  const auto& xs = X.desc.shape;
  const auto& ys = Y.desc.shape;
  if (xs != ys) return Status::ShapeMismatch;

  const int N = (int)xs[0];
  const bool channels_last = a.channels_last;
  const int C = channels_last ? (int)xs[3] : (int)xs[1];
  const int H = channels_last ? (int)xs[1] : (int)xs[2];
  const int W = channels_last ? (int)xs[2] : (int)xs[3];

  if (a.with_affine) {
    if (!gamma || !beta) return Status::MissingInput;
    if (!is1_f32_cuda(*gamma) || !is1_f32_cuda(*beta)) return Status::Invalid;
    if ((int)gamma->desc.shape[0] != C || (int)beta->desc.shape[0] != C) return Status::ShapeMismatch;
  }

  if (!running_mean || !running_var) return Status::MissingInput;
  if (!is1_f32_cuda(*running_mean) || !is1_f32_cuda(*running_var)) return Status::Invalid;
  if ((int)running_mean->desc.shape[0] != C || (int)running_var->desc.shape[0] != C) return Status::ShapeMismatch;

  const bool training = a.training;

  // training 시 save_* 필요
  if (training) {
    if (!save_mean || !save_invstd) return Status::MissingInput;
    if (!is1_f32_cuda(*save_mean) || !is1_f32_cuda(*save_invstd)) return Status::Invalid;
    if ((int)save_mean->desc.shape[0] != C || (int)save_invstd->desc.shape[0] != C) return Status::ShapeMismatch;
  }

  auto s = to_cuda(stream);

  // ---- 경로 분기 ----
  if (training) {
    // 1) 배치 mean/var 계산 (Welford)
    welford_reduce_meanvar_launcher(
      static_cast<const float*>(X.data),
      static_cast<float*>(save_mean->data),
      /*var out*/ static_cast<float*>(running_var->data), // 임시로 var 버퍼 재사용 가능(별도 버퍼 없으면)
      N, C, H, W, channels_last, s);

    // 2) invstd = 1/sqrt(var + eps)
    {
      dim3 block(256), grid((C + 255)/255 + 1);
      compute_invstd_kernel<<<grid, block, 0, s>>>(
        static_cast<const float*>(running_var->data),
        static_cast<float*>(save_invstd->data),
        a.eps, C);
    }

    // 3) Y = (X - mean) * invstd * gamma + beta
    bn_forward_normalize_affine_launcher(
      static_cast<const float*>(X.data),
      static_cast<const float*>(save_mean->data),
      static_cast<const float*>(save_invstd->data),
      a.with_affine ? static_cast<const float*>(gamma->data) : nullptr,
      a.with_affine ? static_cast<const float*>(beta->data)  : nullptr,
      static_cast<float*>(Y.data),
      N, C, H, W, channels_last, s
    );

    // 4) running 통계 갱신 (running_var에는 원래 var가 있어야 하므로, 위에서 var를 running_var에 썼다면 그대로 사용)
    {
      dim3 block(256), grid((C + block.x - 1)/block.x);
      bn_update_running_kernel<<<grid, block, 0, s>>>(
        static_cast<float*>(running_mean->data),
        static_cast<float*>(running_var->data),
        static_cast<const float*>(save_mean->data),
        static_cast<const float*>(running_var->data),
        a.momentum, C
      );
    }
  } else {
    // Inference 경로: running_mean/var로 invstd 만들고 정규화/affine
    if (!running_mean || !running_var) return Status::MissingInput;

    // 임시 invstd를 save_invstd가 있으면 재사용, 없으면 running_var 버퍼를 덮지 않게 주의
    Tensor invstdT;
    bool need_temp_invstd = (save_invstd == nullptr);
    if (need_temp_invstd) {
      // 캡처-세이프 요구로 외부에서 미리 save_invstd를 제공하는 것을 권장합니다.
      // 여기서는 편의상 running_var를 덮지 않도록 별도 경로를 요구(없으면 Invalid).
      return Status::MissingInput; // 캡처/동시성 안전을 위해 invstd 저장 버퍼를 외부에서 주세요.
    }

    // invstd = 1/sqrt(running_var + eps)
    {
      dim3 block(256), grid((C + block.x - 1)/block.x);
      compute_invstd_kernel<<<grid, block, 0, s>>>(
        static_cast<const float*>(running_var->data),
        static_cast<float*>(save_invstd->data),
        a.eps, C);
    }

    // 정규화/affine
    bn_forward_normalize_affine_launcher(
      static_cast<const float*>(X.data),
      static_cast<const float*>(running_mean->data),
      static_cast<const float*>(save_invstd->data),
      a.with_affine ? static_cast<const float*>(gamma->data) : nullptr,
      a.with_affine ? static_cast<const float*>(beta->data)  : nullptr,
      static_cast<float*>(Y.data),
      N, C, H, W, channels_last, s
    );
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
  if (!is4_f32_cuda(dY) || !is4_f32_cuda(X)) return Status::Invalid;
  if (!is1_f32_cuda(save_mean) || !is1_f32_cuda(save_invstd)) return Status::Invalid;

  // shape 일치성
  if (dY.desc.shape != X.desc.shape) return Status::ShapeMismatch;
  if (save_mean.desc.shape[0] != save_invstd.desc.shape[0]) return Status::ShapeMismatch;

  const auto& sX = X.desc.shape;
  const int N = (int)sX[0];
  const bool channels_last = a.channels_last;
  const int C = channels_last ? (int)sX[3] : (int)sX[1];
  const int H = channels_last ? (int)sX[1] : (int)sX[2];
  const int W = channels_last ? (int)sX[2] : (int)sX[3];

  if ((int)save_mean.desc.shape[0] != C) return Status::ShapeMismatch;

  if (dgamma) {
    if (!is1_f32_cuda(*dgamma) || (int)dgamma->desc.shape[0] != C) return Status::ShapeMismatch;
  }
  if (dbeta) {
    if (!is1_f32_cuda(*dbeta) || (int)dbeta->desc.shape[0] != C) return Status::ShapeMismatch;
  }
  if (dX) {
    if (!is4_f32_cuda(*dX) || dX->desc.shape != X.desc.shape) return Status::ShapeMismatch;
  }

  auto s = to_cuda(stream);

  // 1) dgamma, dbeta 감소
  if (dgamma || dbeta) {
    bn_backward_reduce_dbeta_dgamma_launcher(
      static_cast<const float*>(dY.data),
      static_cast<const float*>(X.data),
      static_cast<const float*>(save_mean.data),
      static_cast<const float*>(save_invstd.data),
      dbeta ? static_cast<float*>(dbeta->data) : nullptr,
      dgamma ? static_cast<float*>(dgamma->data) : nullptr,
      N, C, H, W, channels_last, s
    );
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
      N, C, H, W, channels_last, s
    );
  }

  return Status::Ok;
}

} // namespace ai
