// backends/cuda/ops/rnn/kernels.cu
#include <cuda_runtime.h>
#include <math.h>
#include "backends/cuda/ops/rnn/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif

namespace ai {

// ===== device kernels =====
__global__ void kfill_zero(float* p, int64_t n){
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.f;
}

__global__ void kadd_bias_rowwise(float* __restrict__ Y,
                                  const float* __restrict__ b,
                                  int B, int H) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col
  if (i < B && j < H) Y[i*H + j] += b[j];
}

__global__ void kadd_out(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int64_t n){
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}

__global__ void kadd_inplace(float* __restrict__ A,
                             const float* __restrict__ B,
                             int64_t n){
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
}

__global__ void ktanh_out(const float* __restrict__ X,
                          float* __restrict__ Y,
                          int64_t n){
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) Y[i] = tanhf(X[i]);
}

__global__ void ktanh_bwd_from_out(const float* __restrict__ Y,
                                   const float* __restrict__ dY,
                                   float* __restrict__ dZ,
                                   int64_t n){
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float y = Y[i];
    dZ[i] = dY[i] * (1.f - y*y);
  }
}

__global__ void krowwise_sum_accum(const float* __restrict__ M,
                                   float* __restrict__ out,
                                   int B, int H){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  float acc = 0.f;
  #pragma unroll 4
  for (int i=0;i<B;++i) acc += M[i*H + j];
  atomicAdd(out + j, acc);
}

// ===== helpers =====
static inline int div_up(int n, int d){ return (n + d - 1) / d; }
static inline int64_t numel_of(const Tensor& t){ return t.numel(); }

// ===== host wrappers =====
Status fill_zero(Tensor& t, StreamHandle s){
  int64_t n = numel_of(t);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up((int)n, 256));
  kfill_zero<<<gs,bs,0,(cudaStream_t)s>>>(t.data_ptr<float>(), n);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status add_bias_rowwise(Tensor& Y, const Tensor& b, int B, int H, StreamHandle s){
  if (B <= 0 || H <= 0) return Status::Ok;
  dim3 bs(32,8), gs(div_up(H,bs.x), div_up(B,bs.y));
  kadd_bias_rowwise<<<gs,bs,0,(cudaStream_t)s>>>(Y.data_ptr<float>(), b.data_ptr<const float>(), B, H);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status add_out(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s){
  int64_t n = numel_of(A);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up((int)n,256));
  kadd_out<<<gs,bs,0,(cudaStream_t)s>>>(A.data_ptr<const float>(), B.data_ptr<const float>(), C.data_ptr<float>(), n);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status add_inplace(Tensor& A, const Tensor& B, StreamHandle s){
  int64_t n = numel_of(A);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up((int)n,256));
  kadd_inplace<<<gs,bs,0,(cudaStream_t)s>>>(A.data_ptr<float>(), B.data_ptr<const float>(), n);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status tanh_out(const Tensor& X, Tensor& Y, StreamHandle s){
  int64_t n = numel_of(X);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up((int)n,256));
  ktanh_out<<<gs,bs,0,(cudaStream_t)s>>>(X.data_ptr<const float>(), Y.data_ptr<float>(), n);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status tanh_bwd_from_out(const Tensor& Y, const Tensor& dY, Tensor& dZ, StreamHandle s){
  int64_t n = numel_of(Y);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up((int)n,256));
  ktanh_bwd_from_out<<<gs,bs,0,(cudaStream_t)s>>>(Y.data_ptr<const float>(), dY.data_ptr<const float>(), dZ.data_ptr<float>(), n);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

Status rowwise_sum_accum(const Tensor& M, Tensor& out, int B, int H, StreamHandle s){
  if (H <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up(H,256));
  krowwise_sum_accum<<<gs,bs,0,(cudaStream_t)s>>>(M.data_ptr<const float>(), out.data_ptr<float>(), B, H);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
