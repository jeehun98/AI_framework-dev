// backends/cuda/ops/layernorm/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace {  // ← 커널/헬퍼는 TU-local

// warp 합
static __device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

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

  // --- 공유메모리 합 버퍼 ---
  __shared__ float s_sum;

  // 1) mean = sum(x) / N
  if (threadIdx.x == 0) s_sum = 0.0f;
  __syncthreads();

  float sum = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) sum += x[i];

  float wsum = warp_sum(sum);
  if ((threadIdx.x & 31) == 0) atomicAdd(&s_sum, wsum);
  __syncthreads();

  float mean = s_sum / N;

  // 2) var = E[(x-mean)^2]
  if (threadIdx.x == 0) s_sum = 0.0f;
  __syncthreads();

  float vsum = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float d = x[i] - mean;
    vsum += d * d;
  }

  float wvsum = warp_sum(vsum);
  if ((threadIdx.x & 31) == 0) atomicAdd(&s_sum, wvsum);
  __syncthreads();

  float var = s_sum / N;
  float inv_std = rsqrtf(var + eps);

  // 3) normalize + affine
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float norm = (x[i] - mean) * inv_std;
    if (gamma) norm *= gamma[i];
    if (beta)  norm += beta[i];
    y[i] = norm;
  }
}

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

  // 1) sum_x, sum_x2
  float sum_x  = 0.f;
  float sum_x2 = 0.f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float xv = x[i];
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

  float mean    = s_acc[0] / N;
  float var     = s_acc[1] / N - mean * mean;
  float inv_std = rsqrtf(var + eps);

  // 2) sum_dy, sum_dy_xhat  (gy는 gamma 적용 후의 유효 그라드)
  float local_sum_dy    = 0.f;
  float local_sum_dy_xh = 0.f;

  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float xhat = (x[i] - mean) * inv_std;
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

  float sum_dy    = s_acc[2];
  float sum_dy_xh = s_acc[3];

  // 3) dX = (1/N) * inv_std * [ N*gy - sum(gy) - xhat*sum(gy*xhat) ]
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float xhat = (x[i] - mean) * inv_std;
    float gyi  = gy[i];
    if (gamma) gyi *= gamma[i];
    float t = (float)N * gyi - sum_dy - xhat * sum_dy_xh;
    dx[i] = t * (inv_std / N);
  }

  // 4) dgamma/dbeta: 열 방향 합 (전역 누적)
  if (dgamma || dbeta) {
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
      float xhat = (x[i] - mean) * inv_std;
      float gy0  = gy[i]; // 원래 dY (gamma 미적용)
      if (dgamma) atomicAdd(&dgamma[i], gy0 * xhat);
      if (dbeta)  atomicAdd(&dbeta[i],  gy0);
    }
  }
}

} // anonymous namespace

// 퍼블릭 런처만 노출
namespace ai {

void layernorm_forward_kernel_launcher(const float* X,
                                       const float* gamma,
                                       const float* beta,
                                       float* Y,
                                       int M, int N,
                                       float eps,
                                       cudaStream_t stream)
{
  const int BS = 256;
  dim3 grid(M);
  dim3 block(BS);
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
  const int BS = 256;
  dim3 grid(M);
  dim3 block(BS);

  // dgamma/dbeta는 호출 전에 0으로 클리어 필요
  if (dgamma) cudaMemsetAsync(dgamma, 0, sizeof(float) * N, stream);
  if (dbeta)  cudaMemsetAsync(dbeta,  0, sizeof(float) * N, stream);

  layernorm_bwd_kernel<BS><<<grid, block, 0, stream>>>(X, gamma, dY, dX, dgamma, dbeta, M, N, eps);
}

} // namespace ai
