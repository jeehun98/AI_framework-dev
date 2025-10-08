// backends/cuda/ops/rmsnorm/launcher.cu
#include <cuda_runtime.h>
#include "backends/cuda/ops/rmsnorm/api.hpp"

namespace ai {

static inline bool is_rm2d_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// --- kernels (raw) ---
void rmsnorm_forward_kernel_launcher(const float* X,
                                     const float* gamma,
                                     const float* beta,
                                     float* Y,
                                     int M, int N,
                                     float eps,
                                     cudaStream_t s);

void rmsnorm_backward_kernel_launcher(const float* X,
                                      const float* gamma,
                                      const float* dY,
                                      float* dX,
                                      float* dgamma,
                                      float* dbeta,
                                      int M, int N,
                                      float eps,
                                      cudaStream_t s);

// ----------------- Forward -----------------
Status RMSNormCudaLaunch(const Tensor& X,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor& Y,
                         const RMSNormAttrs& attrs,
                         StreamHandle stream)
{
  if (!is_rm2d_f32_cuda(X) || !is_rm2d_f32_cuda(Y)) return Status::Invalid;
  if (X.desc.shape != Y.desc.shape) return Status::ShapeMismatch;

  const int M = (int)X.desc.shape[0];
  const int N = (int)X.desc.shape[1];

  const float* gptr=nullptr; const float* bptr=nullptr;
  if (gamma && gamma->data){
    if (!(gamma->desc.dtype==DType::F32 && gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N))
      return Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }
  if (beta && beta->data){
    if (!(beta->desc.dtype==DType::F32 && beta->desc.shape.size()==1 && beta->desc.shape[0]==N))
      return Status::Invalid;
    bptr = static_cast<const float*>(beta->data);
  }

  rmsnorm_forward_kernel_launcher(
      static_cast<const float*>(X.data), gptr, bptr, static_cast<float*>(Y.data),
      M, N, attrs.eps, to_cuda(stream)
  );
  auto e=cudaPeekAtLastError();
  return (e==cudaSuccess)?Status::Ok:Status::RuntimeError;
}

// ----------------- Backward -----------------
Status RMSNormCudaBackwardLaunch(const Tensor& X,
                                 const Tensor* gamma,
                                 const Tensor& dY,
                                 Tensor& dX,
                                 Tensor* dgamma,
                                 Tensor* dbeta,
                                 const RMSNormAttrs& attrs,
                                 StreamHandle stream)
{
  if (!is_rm2d_f32_cuda(X) || !is_rm2d_f32_cuda(dY) || !is_rm2d_f32_cuda(dX))
    return Status::Invalid;
  if (X.desc.shape != dY.desc.shape || X.desc.shape != dX.desc.shape)
    return Status::ShapeMismatch;

  const int N = (int)X.desc.shape[1];

  const float* gptr=nullptr;
  if (gamma && gamma->data){
    if (!(gamma->desc.dtype==DType::F32 && gamma->desc.shape.size()==1 && gamma->desc.shape[0]==N))
      return Status::Invalid;
    gptr = static_cast<const float*>(gamma->data);
  }

  float* dgamma_ptr=nullptr;
  float* dbeta_ptr =nullptr;
  if (dgamma){
    if (!(dgamma->desc.dtype==DType::F32 && dgamma->desc.shape.size()==1 && dgamma->desc.shape[0]==N))
      return Status::Invalid;
    dgamma_ptr = static_cast<float*>(dgamma->data);
  }
  if (dbeta){
    if (!(dbeta->desc.dtype==DType::F32 && dbeta->desc.shape.size()==1 && dbeta->desc.shape[0]==N))
      return Status::Invalid;
    dbeta_ptr = static_cast<float*>(dbeta->data);
  }

  rmsnorm_backward_kernel_launcher(
      static_cast<const float*>(X.data), gptr, static_cast<const float*>(dY.data),
      static_cast<float*>(dX.data), dgamma_ptr, dbeta_ptr,
      (int)X.desc.shape[0], N, attrs.eps, to_cuda(stream)
  );
  auto e=cudaPeekAtLastError();
  return (e==cudaSuccess)?Status::Ok:Status::RuntimeError;
}

} // namespace ai
