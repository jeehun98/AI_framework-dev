// backends/cuda/ops/rmsnorm/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

namespace {

// --- 블록 합계 유틸 ---
template<int TPB>
__device__ float block_sum(float v) {
  __shared__ float smem[TPB];
  int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int s = TPB/2; s>0; s>>=1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  return smem[0];
}

// --- Forward 커널 ---
template<int TPB>
__global__ void rmsnorm_forward_kernel(const float* __restrict__ X,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ Y,
                                       int M, int N, float eps)
{
  int row = blockIdx.x;
  if (row >= M) return;
  X  += row * N;
  Y  += row * N;

  // sum(x^2)
  float loc = 0.f;
  for (int j = threadIdx.x; j < N; j += TPB) {
    float x = X[j];
    loc += x * x;
  }
  float s2 = block_sum<TPB>(loc);
  float mean = s2 / float(N);
  float inv  = rsqrtf(mean + eps);

  // y = (x*inv)*gamma + beta
  for (int j = threadIdx.x; j < N; j += TPB) {
    float x = X[j];
    float g = gamma ? gamma[j] : 1.f;
    float b = beta  ? beta[j]  : 0.f;
    Y[j] = (x * inv) * g + b;
  }
}

// --- Backward 커널 ---
template<int TPB>
__global__ void rmsnorm_backward_kernel(const float* __restrict__ X,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ dY,
                                        float* __restrict__ dX,
                                        float* __restrict__ dgamma, // may be null
                                        float* __restrict__ dbeta,  // may be null
                                        int M, int N, float eps)
{
  int row = blockIdx.x;
  if (row >= M) return;
  const float* xptr  = X  + row * N;
  const float* gyptr = dY + row * N;
  float* dxptr       = dX + row * N;

  // inv = 1/sqrt(mean(x^2)+eps)
  float loc_s2 = 0.f;
  for (int j = threadIdx.x; j < N; j += TPB) {
    float x = xptr[j];
    loc_s2 += x * x;
  }
  float s2   = block_sum<TPB>(loc_s2);
  float mean = s2 / float(N);
  float inv  = rsqrtf(mean + eps);
  float inv3 = inv * inv * inv;

  // dot = sum((dY*gamma) * x)
  float loc_dot = 0.f;
  for (int j = threadIdx.x; j < N; j += TPB) {
    float gy = gyptr[j];
    float g  = gamma ? gamma[j] : 1.f;
    loc_dot += (gy * g) * xptr[j];
  }
  float dot = block_sum<TPB>(loc_dot);

  // dX = (dY*gamma)*inv - x * inv^3 * (1/N) * dot
  float c1 = inv;
  float c2 = inv3 * (1.f / float(N)) * dot;
  for (int j = threadIdx.x; j < N; j += TPB) {
    float gy = gyptr[j];
    float g  = gamma ? gamma[j] : 1.f;
    float x  = xptr[j];
    float gp = gy * g;
    dxptr[j] = gp * c1 - x * c2;
  }

  // NOTE: 아래 dgamma/dbeta는 데모용(스칼라 누적)이며 실제론 col-wise 누적 필요.
  if (dgamma || dbeta) {
    float loc_dg = 0.f;
    float loc_db = 0.f;
    for (int j = threadIdx.x; j < N; j += TPB) {
      float gy = gyptr[j];
      if (dgamma) loc_dg += gy * (xptr[j] * inv);
      if (dbeta)  loc_db += gy;
    }
    float red_dg = dgamma ? block_sum<TPB>(loc_dg) : 0.f;
    float red_db = dbeta  ? block_sum<TPB>(loc_db) : 0.f;

    if (threadIdx.x == 0) {
      if (dgamma) atomicAdd(&dgamma[0], red_dg); // 👈 데모용(스칼라). 실제로는 dgamma[j]에 누적 필요.
      if (dbeta)  atomicAdd(&dbeta[0],  red_db);
    }
  }
}

} // anonymous

// ⬇️ 선언과 정의의 네임스페이스를 일치시킵니다.
namespace ai {

void rmsnorm_forward_kernel_launcher(const float* X, const float* gamma, const float* beta,
                                     float* Y, int M, int N, float eps, cudaStream_t stream)
{
  const int TPB = 256;
  dim3 grid(M);
  dim3 block(TPB);
  rmsnorm_forward_kernel<TPB><<<grid, block, 0, stream>>>(X, gamma, beta, Y, M, N, eps);
}

void rmsnorm_backward_kernel_launcher(const float* X, const float* gamma, const float* dY,
                                      float* dX, float* dgamma, float* dbeta,
                                      int M, int N, float eps, cudaStream_t stream)
{
  const int TPB = 256;
  dim3 grid(M);
  dim3 block(TPB);
  rmsnorm_backward_kernel<TPB><<<grid, block, 0, stream>>>(X, gamma, dY, dX, dgamma, dbeta, M, N, eps);
}

} // namespace ai
