// backends/cuda/ops/batchnorm/kernels.cu
/* ========================================================================== *
 *  BatchNorm CUDA kernels (NCHW / NHWC)
 *  Files: backends/cuda/ops/batchnorm/kernels.cu
 *
 *  Provided kernels (TU-local):
 *    - reduce_mean_var_kernel<CHANNELS_LAST>
 *        Per-channel Σx, Σx^2 over M = N*H*W (biased var: 1/M).
 *        One CTA per channel. Thread-local FP64 accumulate, block reduce in FP32.
 *
 *    - bn_forward_norm_affine_kernel<CHANNELS_LAST>
 *        y = ((x - mean[c]) * invstd[c]) * gamma[c] + beta[c]   (affine optional)
 *        One CTA per channel; grid-stride loop over M.
 *
 *    - bn_bwd_dbeta_dgamma_kernel<CHANNELS_LAST>
 *        dbeta[c]  += Σ dY
 *        dgamma[c] += Σ dY * x_hat,   x_hat = (x - mean[c]) * invstd[c]
 *        One CTA per channel; caller must zero dbeta/dgamma before launch.
 *
 *    - bn_bwd_dx_kernel<CHANNELS_LAST>
 *        Let dyγ = dY * (gamma or 1), S1 = Σ dyγ, S2 = Σ dyγ * x_hat.
 *        dX = (1/M) * invstd * ( M*dyγ - S1 - x_hat*S2 )
 *        One CTA per channel; two-phase: reduce → write dX.
 *
 *  Launchers (visible symbols in namespace ai):
 *    - welford_reduce_meanvar_launcher         (shmem: 2*BS*sizeof(float))
 *    - bn_forward_normalize_affine_launcher    (shmem: 0)
 *    - bn_backward_reduce_dbeta_dgamma_launcher(shmem: 2*BS*sizeof(float))
 *    - bn_backward_dx_launcher                 (shmem: 2*BS*sizeof(float))
 *
 *  Contracts / Constraints:
 *    - Layout: channels_last==false → NCHW, true → NHWC.
 *    - Types : All kernel pointers are float*. Mixed-precision handled at API/launcher.
 *    - Aliasing: dY/X/Y must not alias when writing.
 *    - Initialization: dbeta/dgamma must be zeroed by caller (+= semantics).
 *    - Epsilon: invstd must be precomputed as rsqrt(var + eps) by the caller.
 *    - Momentum / running_* EMA updates are done in launcher/API, not here.
 *    - Graph capture safe: no dynamic allocation; fixed shmem sizes.
 *
 *  Determinism:
 *    - Fixed one-CTA-per-channel, no atomics → deterministic for given config.
 *
 *  Performance Notes:
 *    - NCHW: for fixed c, contiguous H*W stride → coalesced access.
 *    - NHWC: for fixed c, elements are spaced by C → bandwidth-limited if C is large.
 *    - Block size BS=256 with 2*BS floats of shmem per block (reduce kernels).
 *
 *  Known Limits / TODO:
 *    - “Welford” naming aside, current impl is 1-pass moments (Σx, Σx^2).
 *    - Use int64_t for M to prevent overflow on very large tensors.
 *    - Optional: add NHWC vectorization (float4) when C % 4 == 0 and aligned.
 * ========================================================================== */

#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace { // TU-local device kernels/utilities

// Common: (N,C,H,W) -> per-channel M = N*H*W
__device__ __forceinline__ int64_t spatial_size(int N, int H, int W) {
  return static_cast<int64_t>(N) * H * W;
}

// Indexing helpers
__device__ __forceinline__ size_t idx_nchw(int n,int c,int h,int w,int C,int H,int W){
  return (static_cast<size_t>(n)*C*H*W) + (static_cast<size_t>(c)*H*W) + (static_cast<size_t>(h)*W) + w;
}
__device__ __forceinline__ size_t idx_nhwc(int n,int h,int w,int c,int C,int H,int W){
  return ((static_cast<size_t>(n)*H*W) + (static_cast<size_t>(h)*W) + w) * C + c;
}

// ======================== Forward: mean/var reduce (1 CTA / channel) ========================
template<bool CHANNELS_LAST, int BS>
__global__ __launch_bounds__(BS)
void reduce_mean_var_kernel(
    const float* __restrict__ X,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W)
{
  extern __shared__ float smem[];
  float* s_sum   = smem;            // [BS]
  float* s_sumsq = smem + BS;       // [BS]

  const int c = blockIdx.x; // one block per channel
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int64_t M = spatial_size(N, H, W);

  // 1) thread-local accumulate (FP64 to mitigate cancellation)
  double lsum = 0.0;
  double lsum2= 0.0;

  for (int64_t m = tid; m < M; m += BS) {
    int n = static_cast<int>(m / (H*W));
    int r = static_cast<int>(m % (H*W));
    int h = r / W;
    int w = r % W;

    const size_t off = CHANNELS_LAST
      ? idx_nhwc(n,h,w,c,C,H,W)
      : idx_nchw(n,c,h,w,C,H,W);

    const float x = X[off];
    lsum  += static_cast<double>(x);
    lsum2 += static_cast<double>(x) * static_cast<double>(x);
  }

  s_sum[tid]   = static_cast<float>(lsum);
  s_sumsq[tid] = static_cast<float>(lsum2);
  __syncthreads();

  // 2) intra-block reduce
  for (int offset = BS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid]   += s_sum[tid + offset];
      s_sumsq[tid] += s_sumsq[tid + offset];
    }
    __syncthreads();
  }

  // 3) write (no atomics: 1 block per channel)
  if (tid == 0) {
    const float invM  = 1.f / static_cast<float>(M);
    const float mmean = s_sum[0] * invM;
    float mvar  = s_sumsq[0] * invM - mmean * mmean;   // biased (1/M)
    if (mvar < 0.f) mvar = 0.f;                        // clamp for numeric safety
    mean[c] = mmean;
    var[c]  = mvar;
  }
}

// ======================== Forward: normalize + affine ========================
template<bool CHANNELS_LAST, int BS>
__global__ __launch_bounds__(BS)
void bn_forward_norm_affine_kernel(
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
  const int64_t M = spatial_size(N, H, W);

  const float mu    = mean[c];
  const float istd  = invstd[c];
  const float g     = (with_affine && gamma) ? gamma[c] : 1.f;
  const float b     = (with_affine && beta ) ? beta[c]  : 0.f;

  for (int64_t m = tid; m < M; m += BS) {
    int n = static_cast<int>(m / (H*W));
    int r = static_cast<int>(m % (H*W));
    int h = r / W;
    int w = r % W;

    const size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W)
                                     : idx_nchw(n,c,h,w,C,H,W);

    const float x = X[off];
    float y = (x - mu) * istd;
    if (with_affine) y = y * g + b;
    Y[off] = y;
  }
}

// ======================== Backward: dbeta & dgamma (1 CTA / channel) ========================
template<bool CHANNELS_LAST, int BS>
__global__ __launch_bounds__(BS)
void bn_bwd_dbeta_dgamma_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ dbeta,   // nullable
    float* __restrict__ dgamma,  // nullable
    int N, int C, int H, int W)
{
  extern __shared__ float smem[];
  float* s_db = smem;        // [BS]
  float* s_dg = smem + BS;   // [BS]

  const int c = blockIdx.x;
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int64_t M = spatial_size(N, H, W);

  const float mu   = mean[c];
  const float istd = invstd[c];

  float l_db = 0.f;
  float l_dg = 0.f;

  for (int64_t m = tid; m < M; m += BS) {
    int n = static_cast<int>(m / (H*W));
    int r = static_cast<int>(m % (H*W));
    int h = r / W;
    int w = r % W;

    const size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W)
                                     : idx_nchw(n,c,h,w,C,H,W);

    const float dy = dY[off];
    l_db += dy; // dbeta = Σ dY
    if (dgamma) {
      const float xhat = (X[off] - mu) * istd;
      l_dg += dy * xhat; // dgamma = Σ dY * x_hat
    }
  }

  s_db[tid] = l_db;
  s_dg[tid] = l_dg;
  __syncthreads();

  for (int offset = BS >> 1; offset > 0; offset >>= 1) {
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

// ======================== Backward: dX (1 CTA / channel, 2-phase) ========================
// x_hat = (x - μ) * invstd, M=N*H*W
// dyγ = dY * (gamma or 1)
// S1 = Σ dyγ
// S2 = Σ dyγ * x_hat
// dX = (1/M) * invstd * ( M*dyγ - S1 - x_hat*S2 )
template<bool CHANNELS_LAST, int BS>
__global__ __launch_bounds__(BS)
void bn_bwd_dx_kernel(
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
  float* s_S1 = smem;        // sum(dyγ)
  float* s_S2 = smem + BS;   // sum(dyγ * x_hat)

  const int c = blockIdx.x;
  if (c >= C) return;

  const int tid = threadIdx.x;
  const int64_t M = spatial_size(N, H, W);

  const float mu   = mean[c];
  const float istd = invstd[c];
  const float g    = (with_affine && gamma) ? gamma[c] : 1.f;

  // Phase 1: reduce S1, S2
  float l_S1 = 0.f;
  float l_S2 = 0.f;

  for (int64_t m = tid; m < M; m += BS) {
    int n = static_cast<int>(m / (H*W));
    int r = static_cast<int>(m % (H*W));
    int h = r / W;
    int w = r % W;

    const size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W)
                                     : idx_nchw(n,c,h,w,C,H,W);
    const float x   = X[off];
    const float dy  = dY[off] * g; // dyγ
    const float xhat= (x - mu) * istd;

    l_S1 += dy;
    l_S2 += dy * xhat;
  }

  s_S1[tid] = l_S1;
  s_S2[tid] = l_S2;
  __syncthreads();

  for (int offset = BS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_S1[tid] += s_S1[tid + offset];
      s_S2[tid] += s_S2[tid + offset];
    }
    __syncthreads();
  }

  const float S1 = s_S1[0];
  const float S2 = s_S2[0];

  // Phase 2: write dX
  for (int64_t m = tid; m < M; m += BS) {
    int n = static_cast<int>(m / (H*W));
    int r = static_cast<int>(m % (H*W));
    int h = r / W;
    int w = r % W;

    const size_t off = CHANNELS_LAST ? idx_nhwc(n,h,w,c,C,H,W)
                                     : idx_nchw(n,c,h,w,C,H,W);

    const float x   = X[off];
    const float dy  = dY[off] * g; // dyγ
    const float xhat= (x - mu) * istd;

    const float fM  = static_cast<float>(M);
    const float dx  = (1.f / fM) * istd * (fM * dy - S1 - xhat * S2);
    dX[off] = dx;
  }
}

} // anonymous namespace


namespace ai { // ---- visible symbols: launchers ----

static constexpr int BS = 256;

// (1) mean/var 감소 런처
void welford_reduce_meanvar_launcher(
    const float* X, float* mean, float* var,
    int N, int C, int H, int W, bool channels_last, cudaStream_t s)
{
  dim3 block(BS), grid(C);
  const size_t shmem = BS * sizeof(float) * 2; // sum, sumsq
  if (channels_last) {
    reduce_mean_var_kernel<true,  BS><<<grid, block, shmem, s>>>(X, mean, var, N, C, H, W);
  } else {
    reduce_mean_var_kernel<false, BS><<<grid, block, shmem, s>>>(X, mean, var, N, C, H, W);
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
  dim3 block(BS), grid(C);
  const bool with_affine = (gamma != nullptr) || (beta != nullptr);
  if (channels_last) {
    bn_forward_norm_affine_kernel<true,  BS><<<grid, block, 0, s>>>(
      X, mean, invstd, gamma, beta, Y, N, C, H, W, with_affine);
  } else {
    bn_forward_norm_affine_kernel<false, BS><<<grid, block, 0, s>>>(
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
  dim3 block(BS), grid(C);
  const size_t shmem = BS * sizeof(float) * 2; // dbeta, dgamma partials
  if (channels_last) {
    bn_bwd_dbeta_dgamma_kernel<true,  BS><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, dbeta, dgamma, N, C, H, W);
  } else {
    bn_bwd_dbeta_dgamma_kernel<false, BS><<<grid, block, shmem, s>>>(
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
  dim3 block(BS), grid(C);
  const size_t shmem = BS * sizeof(float) * 2; // S1, S2
  const bool with_affine = (gamma != nullptr);
  if (channels_last) {
    bn_bwd_dx_kernel<true,  BS><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, gamma, dX, N, C, H, W, with_affine);
  } else {
    bn_bwd_dx_kernel<false, BS><<<grid, block, shmem, s>>>(
      dY, X, mean, invstd, gamma, dX, N, C, H, W, with_affine);
  }
}

} // namespace ai
