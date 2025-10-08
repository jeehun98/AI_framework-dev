// backends/cuda/ops/rmsnorm/kernels.cu
#include <cuda_runtime.h>
#include <cmath>

namespace {

static __device__ __forceinline__ float warp_sum(float v){
  for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

template<int BS>
__global__ void rmsnorm_fwd_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ Y,
                                   int M, int N, float eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * N;
  float*       y = Y + row * N;

  // mean of squares
  __shared__ float s_sq;
  if (threadIdx.x==0) s_sq=0.f;
  __syncthreads();

  float acc=0.f;
  for (int i=threadIdx.x;i<N;i+=BS) { float v=x[i]; acc += v*v; }
  float w=warp_sum(acc);
  if ((threadIdx.x & 31)==0) atomicAdd(&s_sq, w);
  __syncthreads();

  const float rms = sqrtf(s_sq / (float)N + eps);
  const float inv_rms = 1.f / rms;

  for (int i=threadIdx.x;i<N;i+=BS){
    float n = x[i] * inv_rms;
    if (gamma) n *= gamma[i];
    if (beta)  n += beta[i];
    y[i] = n;
  }
}

template<int BS>
__global__ void rmsnorm_bwd_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ dY,
                                   float* __restrict__ dX,
                                   float* __restrict__ dgamma,
                                   float* __restrict__ dbeta,
                                   int M, int N, float eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x  = X  + row * N;
  const float* gy = dY + row * N;
  float*       dx = dX + row * N;

  // 1) sum(x^2), sum(gy * x * (gamma?))
  __shared__ float s_sum_x2, s_sum_gyxg;
  if (threadIdx.x==0){ s_sum_x2=0.f; s_sum_gyxg=0.f; }
  __syncthreads();

  float local_x2 = 0.f;
  float local_gyxg = 0.f;

  for (int i=threadIdx.x;i<N;i+=BS){
    float g = gamma ? gamma[i] : 1.f;
    float xv = x[i];
    local_x2   += xv*xv;
    local_gyxg += gy[i] * xv * g;
  }

  float w1 = warp_sum(local_x2);
  float w2 = warp_sum(local_gyxg);
  if ((threadIdx.x & 31)==0){
    atomicAdd(&s_sum_x2,  w1);
    atomicAdd(&s_sum_gyxg,w2);
  }
  __syncthreads();

  const float rms     = sqrtf(s_sum_x2 / (float)N + eps);
  const float inv_rms = 1.f / rms;

  // 2) dX
  // y = (x / rms) * g + b → dy/dx = g*(1/rms) + (x)*d(1/rms)/dx
  // 더 간단한 표준 결과: dX = g * ( dY * (1/rms) - x * (sum(dY*g*x)/ (N * rms^3)) )
  const float coef = s_sum_gyxg / ((float)N * rms * rms * rms);

  for (int i=threadIdx.x;i<N;i+=BS){
    float g = gamma ? gamma[i] : 1.f;
    float xv = x[i];
    dx[i] = g * ( gy[i] * inv_rms - xv * coef );
  }

  // 3) dgamma/dbeta (row-wise 누적 → 전역 원소별 합)
  if (dgamma || dbeta){
    for (int i=threadIdx.x;i<N;i+=BS){
      float xhat = x[i] * inv_rms;
      if (dgamma) atomicAdd(&dgamma[i], gy[i] * xhat);
      if (dbeta)  atomicAdd(&dbeta[i],  gy[i]);
    }
  }
}

} // anon

namespace ai {

void rmsnorm_forward_kernel_launcher(const float* X,
                                     const float* gamma,
                                     const float* beta,
                                     float* Y,
                                     int M, int N,
                                     float eps,
                                     cudaStream_t s)
{
  constexpr int BS=256;
  dim3 grid(M), block(BS);
  rmsnorm_fwd_kernel<BS><<<grid, block, 0, s>>>(X, gamma, beta, Y, M, N, eps);
}

void rmsnorm_backward_kernel_launcher(const float* X,
                                      const float* gamma,
                                      const float* dY,
                                      float* dX,
                                      float* dgamma,
                                      float* dbeta,
                                      int M, int N,
                                      float eps,
                                      cudaStream_t s)
{
  constexpr int BS=256;
  dim3 grid(M), block(BS);

  // dgamma/dbeta는 호출 전 0으로 초기화 필요
  if (dgamma) cudaMemsetAsync(dgamma, 0, sizeof(float)*N, s);
  if (dbeta)  cudaMemsetAsync(dbeta,  0, sizeof(float)*N, s);

  rmsnorm_bwd_kernel<BS><<<grid, block, 0, s>>>(X, gamma, dY, dX, dgamma, dbeta, M, N, eps);
}

} // namespace ai
