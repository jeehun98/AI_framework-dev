#include "api.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include "ai/dispatch.hpp"

namespace ai {

// 커널 선언
void rmsnorm_forward_kernel_launcher(const float* X, const float* gamma, const float* beta,
                                     float* Y, int M, int N, float eps, cudaStream_t stream);
void rmsnorm_backward_kernel_launcher(const float* X, const float* gamma, const float* dY,
                                      float* dX, float* dgamma, float* dbeta,
                                      int M, int N, float eps, cudaStream_t stream);

static inline bool is_row_major_2d_f32(const Tensor& T) {
  return T.desc.dtype==DType::F32 && T.desc.layout==Layout::RowMajor && T.desc.shape.size()==2;
}

ai::Status RMSNormCudaLaunch(const Tensor& X,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor& Y,
                         const RMSNormAttrs& attrs,
                         StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y)) return ai::Status::Invalid;
  if (X.device!=Device::CUDA || Y.device!=Device::CUDA)    return ai::Status::Invalid;
  if (X.desc.shape!=Y.desc.shape)                          return ai::Status::Invalid;
  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const float* gptr = nullptr;
  const float* bptr = nullptr;
  if (gamma && gamma->data) {
    if (!(gamma->desc.dtype==DType::F32 && gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N)) return Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }
  if (beta && beta->data) {
    if (!(beta->desc.dtype==DType::F32 && beta->desc.shape.size()==1 && beta->desc.shape[0]==N)) return Status::Invalid;
    bptr = static_cast<const float*>(beta->data);
  }

  auto* xptr = static_cast<const float*>(X.data);
  auto* yptr = static_cast<float*>(Y.data);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  rmsnorm_forward_kernel_launcher(xptr, gptr, bptr, yptr, M, N, attrs.eps, s);
  return ai::Status::Ok;
}

ai::Status RMSNormCudaBackwardLaunch(const Tensor& X,
                                 const Tensor* gamma,
                                 const Tensor& dY,
                                 Tensor& dX,
                                 Tensor* dgamma,
                                 Tensor* dbeta,
                                 const RMSNormAttrs& attrs,
                                 StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX)) return ai::Status::Invalid;
  if (X.device!=Device::CUDA || dY.device!=Device::CUDA || dX.device!=Device::CUDA)      return ai::Status::Invalid;
  if (X.desc.shape!=dY.desc.shape || X.desc.shape!=dX.desc.shape)                        return ai::Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const float* gptr = nullptr;
  if (gamma && gamma->data) {
    if (!(gamma->desc.dtype==DType::F32 && gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N)) return ai::Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }

  float* dgamma_ptr = nullptr;
  float* dbeta_ptr  = nullptr;
  if (dgamma) {
    if (!(dgamma->desc.dtype==DType::F32 && dgamma->desc.shape.size()==1 && dgamma->desc.shape[0]==N)) return ai::Status::Invalid;
    dgamma_ptr = static_cast<float*>(dgamma->data);
  }
  if (dbeta) {
    if (!(dbeta->desc.dtype==DType::F32 && dbeta->desc.shape.size()==1 && dbeta->desc.shape[0]==N)) return ai::Status::Invalid;
    dbeta_ptr = static_cast<float*>(dbeta->data);
  }

  auto* xptr  = static_cast<const float*>(X.data);
  auto* dyptr = static_cast<const float*>(dY.data);
  auto* dxptr = static_cast<float*>(dX.data);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  rmsnorm_backward_kernel_launcher(xptr, gptr, dyptr, dxptr, dgamma_ptr, dbeta_ptr, M, N, attrs.eps, s);
  return ai::Status::Ok;
}

} // namespace ai
