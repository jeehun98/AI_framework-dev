// backends/cuda/ops/layernorm/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace {  // TU-local

// ----- warp reduction helpers -----
static __device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

/**
 * Forward kernel
 *  - Row-wise LayerNorm over N
 *  - Optional affine (gamma, beta)
 *  - No dynamic allocations; capture-safe
 */
template<int BLOCK_SIZE>
__global__ void layernorm_fwd_kernel(const float* __restrict__ X,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ Y,
                                     int M, int N, float eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * N;
  float*       y = Y + row * N;

  __shared__ float s_sum;

  // 1) mean = sum(x) / N
  if (threadIdx.x == 0) s_sum = 0.0f;
  __syncthreads();

  float sum = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) sum += x[i];

  // reduce within warp then atomically accumulate into shared
  float wsum = warp_sum(sum);
  if ((threadIdx.x & 31) == 0) atomicAdd(&s_sum, wsum);
  __syncthreads();

  const float mean = s_sum / (float)N;

  // 2) var = E[(x-mean)^2]
  if (threadIdx.x == 0) s_sum = 0.0f;
  __syncthreads();

  float vsum = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    const float d = x[i] - mean;
    vsum += d * d;
  }

  float wvsum = warp_sum(vsum);
  if ((threadIdx.x & 31) == 0) atomicAdd(&s_sum, wvsum);
  __syncthreads();

  const float var     = s_sum / (float)N;
  const float inv_std = rsqrtf(var + eps);

  // 3) normalize + optional affine
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float out = (x[i] - mean) * inv_std;
    if (gamma) out *= gamma[i];
    if (beta)  out += beta[i];
    y[i] = out;
  }
}

/**
 * Backward kernel
 *  - Computes dX, and optionally dgamma/dbeta (atomic accumulation)
 *  - gamma may be null (scale-free LN)
 *  - dgamma/dbeta pointers may be null (skip those paths)
 *  - No dynamic allocations; capture-safe
 *
 * dX formula (row-wise):
 *   let xhat = (x - mean) * inv_std
 *   let gyi  = dY * gamma (or dY if gamma is null)
 *   sum_dy    = Σ_j gyi
 *   sum_dy_xh = Σ_j gyi * xhat
 *   dX_i = (1/N) * inv_std * [ N*gyi - sum_dy - xhat_i * sum_dy_xh ]
 */
template<int BLOCK_SIZE>
__global__ void layernorm_bwd_kernel(const float* __restrict__ X,
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

  // s_acc: [sum_x, sum_x2, sum_dy, sum_dy_xhat]
  __shared__ float s_acc[4];
  if (threadIdx.x < 4) s_acc[threadIdx.x] = 0.f;
  __syncthreads();

  // 1) sum_x, sum_x2 (for mean/var)
  float sum_x  = 0.f;
  float sum_x2 = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    const float xv = x[i];
    sum_x  += xv;
    sum_x2 += xv * xv;
  }
  float w1 = warp_sum(sum_x);
  float w2 = warp_sum(sum_x2);
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&s_acc[0], w1);
    atomicAdd(&s_acc[1], w2);
  }
  __syncthreads();

  const float mean    = s_acc[0] / (float)N;
  const float var     = s_acc[1] / (float)N - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  // 2) sum_dy, sum_dy_xhat  (gy is already scaled by gamma if present)
  float local_sum_dy    = 0.f;
  float local_sum_dy_xh = 0.f;

  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    const float xhat = (x[i] - mean) * inv_std;
    float gyi  = gy[i];
    if (gamma) gyi *= gamma[i];
    local_sum_dy    += gyi;
    local_sum_dy_xh += gyi * xhat;
  }
  float w3 = warp_sum(local_sum_dy);
  float w4 = warp_sum(local_sum_dy_xh);
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&s_acc[2], w3);
    atomicAdd(&s_acc[3], w4);
  }
  __syncthreads();

  const float sum_dy    = s_acc[2];
  const float sum_dy_xh = s_acc[3];

  // 3) dX
  const float invN = 1.0f / (float)N;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    const float xhat = (x[i] - mean) * inv_std;
    float gyi  = gy[i];
    if (gamma) gyi *= gamma[i];
    const float t = (float)N * gyi - sum_dy - xhat * sum_dy_xh;
    dx[i] = t * (inv_std * invN);
  }

  // 4) dgamma/dbeta: accumulate over rows (global atomics into [N])
  //    NOTE: caller (launcher) must zero-initialize dgamma/dbeta before invoking this kernel.
  if (dgamma || dbeta) {
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
      const float xhat = (x[i] - mean) * inv_std;
      const float gy0  = gy[i]; // raw dY (no gamma scaling)
      if (dgamma) atomicAdd(&dgamma[i], gy0 * xhat);
      if (dbeta)  atomicAdd(&dbeta[i],  gy0);
    }
  }
}

} // anonymous namespace

// Public launchers
namespace ai {

void layernorm_forward_kernel_launcher(const float* X,
                                       const float* gamma,
                                       const float* beta,
                                       float* Y,
                                       int M, int N,
                                       float eps,
                                       cudaStream_t stream)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  layernorm_fwd_kernel<BS><<<grid, block, 0, stream>>>(X, gamma, beta, Y, M, N, eps);
}

void layernorm_backward_kernel_launcher(const float* X,
                                        const float* gamma,
                                        const float* dY,
                                        float* dX,
                                        float* dgamma,
                                        float* dbeta,
                                        int M, int N,
                                        float eps,
                                        cudaStream_t stream)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);

  // If dgamma/dbeta are requested, they must be zeroed before atomic accumulation.
  if (dgamma) cudaMemsetAsync(dgamma, 0, sizeof(float) * (size_t)N, stream);
  if (dbeta)  cudaMemsetAsync(dbeta,  0, sizeof(float) * (size_t)N, stream);

  layernorm_bwd_kernel<BS><<<grid, block, 0, stream>>>(X, gamma, dY, dX, dgamma, dbeta, M, N, eps);
}

} // namespace ai
