// backends/cuda/ops/batchnorm/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace { // TU-local device kernels/utilities

// 공통: (N,C,H,W)에서 채널당 M = N*H*W 원소 수
__device__ __forceinline__ int spatial_size(int N, int H, int W) {
  return N * H * W;
}

// 인덱싱 유틸
__device__ __forceinline__ size_t idx_nchw(int n,int c,int h,int w,int C,int H,int W){
  return ((size_t)n*C*H*W) + ((size_t)c*H*W) + ((size_t)h*W) + w;
}
__device__ __forceinline__ size_t idx_nhwc(int n,int h,int w,int c,int C,int H,int W){
  return (((size_t)n*H*W) + ((size_t)h*W) + w) * C + c;
}

// ======================== Forward: mean/var 감소 (채널당 1 CTA) ========================
template<bool CHANNELS_LAST>
__global__ void reduce_mean_var_kernel(
    const float* __restrict__ X, float* __restrict__ mean, float* __restrict__ var,
    int N, int C, int H, int W)
{
  extern __shared__ float smem[];
  float* s_sum  = smem;                 // [blockDim.x]
  float* s_sumsq= smem + blockDim.x;    // [blockDim.x]

  const int c = blockIdx.x; // one block per channel
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int M = N * H * W;

  // 1) thread-local accumulate
  double lsum = 0.0;
  double lsum2= 0.0;

  for (int m = tid; m < M; m += blockDim.x) {
    // m -> (n,h,w)
    int n = m / (H*W);
    int r = m % (H*W);
    int h = r / W;
    int w = r % W;

    float x = CHANNELS_LAST
      ? X[idx_nhwc(n,h,w,c,C,H,W)]
      : X[idx_nchw(n,c,h,w,C,H,W)];

    lsum  += (double)x;
    lsum2 += (double)x * (double)x;
  }

  s_sum[tid]   = (float)lsum;
  s_sumsq[tid] = (float)lsum2;
  __syncthreads();

  // 2) intra-block reduce
  for (int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid]   += s_sum[tid + offset];
      s_sumsq[tid] += s_sumsq[tid + offset];
    }
    __syncthreads();
  }

  // 3) write out (block 1개이므로 원자연산 불필요)
  if (tid == 0) {
    float sum   = s_sum[0];
    float sumsq = s_sumsq[0];
    float mmean = sum / (float)M;
    // 표본분산(unbiased) 대신 batch 통계(모멘트) 사용 -> 대부분 프레임워크는 biased(1/M) 사용
    float mvar  = sumsq / (float)M - mmean * mmean;
    mean[c] = mmean;
    var[c]  = mvar;
  }
}

// ======================== Forward: normalize + affine ========================
template<bool CHANNELS_LAST>
__global__ void bn_forward_norm_affine_kernel(
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ gamma, // nullable if !affine
    const float* __restrict__ beta,  // nullable if !affine
    float* __restrict__ Y,
    int N, int C, int H, int W,
    bool with_affine)
{
  const int c = blockIdx.x; // one block per channel
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int M = N * H * W;

  const float mu    = mean[c];
  const float istd  = invstd[c];
  const float g     = with_affine && gamma ? gamma[c] : 1.f;
  const float b     = with_affine && beta  ? beta[c]  : 0.f;

  for (int m = tid; m < M; m += blockDim.x) {
    int n = m / (H*W);
    int r = m % (H*W);
    int h = r / W;
    int w = r % W;

    size_t offX = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W) : idx_nchw(n,c,h,w,C,H,W);
    float x = X[offX];
    float y = (x - mu) * istd;
    if (with_affine) y = y * g + b;
    Y[offX] = y;
  }
}

// ======================== Backward: dbeta & dgamma (채널당 1 CTA) ========================
template<bool CHANNELS_LAST>
__global__ void bn_bwd_dbeta_dgamma_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ dbeta,   // nullable
    float* __restrict__ dgamma,  // nullable
    int N, int C, int H, int W)
{
  extern __shared__ float smem[];
  float* s_db = smem;                 // [blockDim.x]
  float* s_dg = smem + blockDim.x;    // [blockDim.x]

  const int c = blockIdx.x;
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int M = N * H * W;

  const float mu   = mean[c];
  const float istd = invstd[c];

  float l_db = 0.f;
  float l_dg = 0.f;

  for (int m = tid; m < M; m += blockDim.x) {
    int n = m / (H*W);
    int r = m % (H*W);
    int h = r / W;
    int w = r % W;

    size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W) : idx_nchw(n,c,h,w,C,H,W);
    float dy = dY[off];
    l_db += dy; // dbeta = Σ dY
    if (dgamma) {
      float xhat = (X[off] - mu) * istd;
      l_dg += dy * xhat; // dgamma = Σ dY * x_hat
    }
  }

  s_db[tid] = l_db;
  s_dg[tid] = l_dg;
  __syncthreads();

  for (int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_db[tid] += s_db[tid + offset];
      s_dg[tid] += s_dg[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (dbeta)  dbeta[c]  += s_db[0];
    if (dgamma) dgamma[c] += s_dg[0];
  }
}

// ======================== Backward: dX (채널당 1 CTA, 2-phase) ========================
// 공식: x_hat = (x - μ) * invstd, M=N*H*W
// dyγ = dY * (gamma or 1)
// S1 = Σ dyγ
// S2 = Σ dyγ * x_hat
// dX = (1/M) * invstd * ( M*dyγ - S1 - x_hat*S2 )
template<bool CHANNELS_LAST>
__global__ void bn_bwd_dx_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ gamma, // nullable if !affine
    float* __restrict__ dX,
    int N, int C, int H, int W,
    bool with_affine)
{
  extern __shared__ float smem[];
  float* s_S1 = smem;                 // sum(dyγ)
  float* s_S2 = smem + blockDim.x;    // sum(dyγ * x_hat)

  const int c = blockIdx.x;
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int M = N * H * W;

  const float mu   = mean[c];
  const float istd = invstd[c];
  const float g    = (with_affine && gamma) ? gamma[c] : 1.f;

  // Phase 1: sums
  float l_S1 = 0.f;
  float l_S2 = 0.f;

  for (int m = tid; m < M; m += blockDim.x) {
    int n = m / (H*W);
    int r = m % (H*W);
    int h = r / W;
    int w = r % W;

    size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W) : idx_nchw(n,c,h,w,C,H,W);
    float x  = X[off];
    float dy = dY[off] * g; // dyγ
    float xhat = (x - mu) * istd;
    l_S1 += dy;
    l_S2 += dy * xhat;
  }

  s_S1[tid] = l_S1;
  s_S2[tid] = l_S2;
  __syncthreads();

  for (int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_S1[tid] += s_S1[tid + offset];
      s_S2[tid] += s_S2[tid + offset];
    }
    __syncthreads();
  }

  float S1 = s_S1[0];
  float S2 = s_S2[0];

  // Phase 2: write dX
  for (int m = tid; m < M; m += blockDim.x) {
    int n = m / (H*W);
    int r = m % (H*W);
    int h = r / W;
    int w = r % W;

    size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W) : idx_nchw(n,c,h,w,C,H,W);
    float x  = X[off];
    float dy = dY[off] * g; // dyγ
    float xhat = (x - mu) * istd;

    float dx = (1.f / (float)M) * istd * ( (float)M * dy - S1 - xhat * S2 );
    dX[off] = dx;
  }
}

} // anonymous namespace


namespace ai { // ---- visible symbols: launchers ----

// (1) mean/var 감소 런처
void welford_reduce_meanvar_launcher(
    const float* X, float* mean, float* var,
    int N, int C, int H, int W, bool channels_last, cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 block(BS), grid(C);
  size_t shmem = BS * sizeof(float) * 2; // sum, sumsq
  if (channels_last) {
    reduce_mean_var_kernel<true> <<<grid, block, shmem, s>>>(X, mean, var, N, C, H, W);
  } else {
    reduce_mean_var_kernel<false><<<grid, block, shmem, s>>>(X, mean, var, N, C, H, W);
  }
}

// (2) 정규화 + affine 런처
void bn_forward_normalize_affine_launcher(
    const float* X,
    const float* mean, const float* invstd,
    const float* gamma, const float* beta,
    float* Y,
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 block(BS), grid(C);
  // with_affine 여부는 gamma/beta 포인터 존재로 판단 (런처에서는 모든 케이스 허용)
  bool with_affine = (gamma != nullptr) || (beta != nullptr);
  if (channels_last) {
    bn_forward_norm_affine_kernel<true><<<grid, block, 0, s>>>(
      X, mean, invstd, gamma, beta, Y, N, C, H, W, with_affine);
  } else {
    bn_forward_norm_affine_kernel<false><<<grid, block, 0, s>>>(
      X, mean, invstd, gamma, beta, Y, N, C, H, W, with_affine);
  }
}

// (3) dgamma/dbeta 감소 런처
void bn_backward_reduce_dbeta_dgamma_launcher(
    const float* dY, const float* X,
    const float* mean, const float* invstd,
    float* dbeta, float* dgamma,
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 block(BS), grid(C);
  size_t shmem = BS * sizeof(float) * 2; // dbeta, dgamma part
  if (channels_last) {
    bn_bwd_dbeta_dgamma_kernel<true><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, dbeta, dgamma, N, C, H, W);
  } else {
    bn_bwd_dbeta_dgamma_kernel<false><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, dbeta, dgamma, N, C, H, W);
  }
}

// (4) dX 런처
void bn_backward_dx_launcher(
    const float* dY, const float* X,
    const float* mean, const float* invstd,
    const float* gamma,
    float* dX,
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 block(BS), grid(C);
  size_t shmem = BS * sizeof(float) * 2; // S1, S2
  // with_affine 여부는 gamma의 null 여부로 판단
  bool with_affine = (gamma != nullptr);
  if (channels_last) {
    bn_bwd_dx_kernel<true><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, gamma, dX, N, C, H, W, with_affine);
  } else {
    bn_bwd_dx_kernel<false><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, gamma, dX, N, C, H, W, with_affine);
  }
}

} // namespace ai
