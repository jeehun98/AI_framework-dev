// backends/cuda/ops/layernorm/launcher.cu
#include <cuda_runtime.h>
#include <cassert>
#include "backends/cuda/ops/layernorm/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

// ── 커널 런처 선언 (CUDA 내부 전용; 원시 포인터 + cudaStream_t) ──
void layernorm_forward_kernel_launcher(const float* X,
                                       const float* gamma,
                                       const float* beta,
                                       float* Y,
                                       int M, int N,
                                       float eps,
                                       cudaStream_t stream);

void layernorm_backward_kernel_launcher(const float* X,
                                        const float* gamma,
                                        const float* dY,
                                        float* dX,
                                        float* dgamma,
                                        float* dbeta,
                                        int M, int N,
                                        float eps,
                                        cudaStream_t stream);

// ── 공용 검증 헬퍼 ──
static inline bool is_row_major_2d_f32(const Tensor& t) {
  return t.desc.dtype  == DType::F32 &&
         t.desc.layout == Layout::RowMajor &&
         t.desc.shape.size() == 2 &&
         t.device == Device::CUDA;
}

static inline cudaStream_t to_cuda(StreamHandle h) {
  return reinterpret_cast<cudaStream_t>(h);
}

// ── Forward ────────────────────────────────────────────────────────────────
Status LayerNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,
                           const Tensor* beta,
                           Tensor& Y,
                           const LayerNormAttrs& attrs,
                           StreamHandle stream,
                           const LayerNormWorkspaceFwd* /*ws_fwd*/)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y)) return Status::Invalid;
  if (X.desc.shape != Y.desc.shape)                        return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  // gamma / beta 검증 (있으면 1D[N])
  const float* gptr = nullptr;
  const float* bptr = nullptr;

  if (gamma && gamma->data) {
    if (!(gamma->desc.dtype==DType::F32 &&
          gamma->desc.layout==Layout::RowMajor &&
          gamma->desc.shape.size()==1 &&
          gamma->desc.shape[0]==N))
      return Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }
  if (beta && beta->data) {
    if (!(beta->desc.dtype==DType::F32 &&
          beta->desc.layout==Layout::RowMajor &&
          beta->desc.shape.size()==1 &&
          beta->desc.shape[0]==N))
      return Status::Invalid;
    bptr = static_cast<const float*>(beta->data);
  }

  // 포인터/스트림
  const float* xptr = static_cast<const float*>(X.data);
  float*       yptr = static_cast<float*>(Y.data);
  cudaStream_t s    = to_cuda(stream);

  // 커널 런처 호출
  layernorm_forward_kernel_launcher(xptr, gptr, bptr, yptr, M, N, attrs.eps, s);

  // CUDA 에러 체크 (캡처-세이프)
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// ── Backward ───────────────────────────────────────────────────────────────
Status LayerNormCudaBackwardLaunch(const Tensor& X,
                                   const Tensor* gamma,
                                   const Tensor& dY,
                                   Tensor& dX,
                                   Tensor* dgamma,
                                   Tensor* dbeta,
                                   const LayerNormAttrs& attrs,
                                   StreamHandle stream,
                                   const LayerNormWorkspaceBwd* /*ws_bwd*/) 
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX))
    return Status::Invalid;
  if (X.desc.shape != dY.desc.shape || X.desc.shape != dX.desc.shape)
    return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  // gamma (있으면 1D[N]) — dX 계산 시 scale 항에 사용
  const float* gptr = nullptr;
  if (gamma && gamma->data) {
    if (!(gamma->desc.dtype==DType::F32 &&
          gamma->desc.layout==Layout::RowMajor &&
          gamma->desc.shape.size()==1 &&
          gamma->desc.shape[0]==N))
      return Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }

  // dgamma/dbeta (있으면 1D[N])
  float* dgamma_ptr = nullptr;
  float* dbeta_ptr  = nullptr;

  if (dgamma) {
    if (!(dgamma->desc.dtype==DType::F32 &&
          dgamma->desc.layout==Layout::RowMajor &&
          dgamma->desc.shape.size()==1 &&
          dgamma->desc.shape[0]==N))
      return Status::Invalid;
    dgamma_ptr = static_cast<float*>(dgamma->data);
  }
  if (dbeta) {
    if (!(dbeta->desc.dtype==DType::F32 &&
          dbeta->desc.layout==Layout::RowMajor &&
          dbeta->desc.shape.size()==1 &&
          dbeta->desc.shape[0]==N))
      return Status::Invalid;
    dbeta_ptr = static_cast<float*>(dbeta->data);
  }

  const float* xptr  = static_cast<const float*>(X.data);
  const float* dyptr = static_cast<const float*>(dY.data);
  float*       dxptr = static_cast<float*>(dX.data);
  cudaStream_t s     = to_cuda(stream);

  // (옵션) 커널이 dgamma/dbeta를 atomicAdd로 누적하는 구현이라면 0 초기화 필요
  // if (dgamma_ptr) cudaMemsetAsync(dgamma_ptr, 0, sizeof(float)*N, s);
  // if (dbeta_ptr)  cudaMemsetAsync(dbeta_ptr,  0, sizeof(float)*N, s);

  layernorm_backward_kernel_launcher(xptr, gptr, dyptr, dxptr,
                                     dgamma_ptr, dbeta_ptr,
                                     M, N, attrs.eps, s);

  // CUDA 에러 체크
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
