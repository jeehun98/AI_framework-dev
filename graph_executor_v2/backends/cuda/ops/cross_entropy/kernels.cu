#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace {

// warp helpers
__device__ __forceinline__ float warp_max(float v){
  for (int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v){
  for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

template<int BS>
__global__ void ce_fwd_logits_kernel(const float* __restrict__ X, // [M,N]
                                     const int32_t* __restrict__ T,// [M]
                                     float* __restrict__ loss_vec, // [M]
                                     int M, int N,
                                     int ignore_index,
                                     float ls_eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  if (t == ignore_index) {
    if (threadIdx.x == 0) loss_vec[row] = 0.f;
    return;
  }
  const float* x = X + row * N;

  // max
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < N; i += BS) local_max = fmaxf(local_max, x[i]);
  float wmax = warp_max(local_max);
  __shared__ float warp_buf_max[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_buf_max[warp_id] = wmax;
  __syncthreads();

  float row_max = -CUDART_INF_F;
  if (warp_id == 0) {
    float tmax = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
    tmax = warp_max(tmax);
    if (threadIdx.x == 0) row_max = tmax;
  }
  __syncthreads();
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = row_max;
  __syncthreads();
  row_max = s_max;

  float local_e = 0.f, local_sumx = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    local_e    += __expf(x[i] - row_max);
    local_sumx += x[i];
  }
  float we  = warp_sum(local_e);
  float wsx = warp_sum(local_sumx);
  __shared__ float warp_buf_e[BS/32], warp_buf_sx[BS/32];
  if ((threadIdx.x & 31) == 0) { warp_buf_e[warp_id] = we; warp_buf_sx[warp_id] = wsx; }
  __syncthreads();

  float denom = 0.f, sumx = 0.f;
  if (warp_id == 0) {
    float te  = (threadIdx.x < (BS/32)) ? warp_buf_e[threadIdx.x]  : 0.f;
    float tsx = (threadIdx.x < (BS/32)) ? warp_buf_sx[threadIdx.x] : 0.f;
    te  = warp_sum(te);
    tsx = warp_sum(tsx);
    if (threadIdx.x == 0) { denom = te; sumx = tsx; }
  }
  __syncthreads();
  __shared__ float s_den, s_sumx;
  if (threadIdx.x == 0) { s_den = denom; s_sumx = sumx; }
  __syncthreads();
  denom = s_den; sumx = s_sumx;

  if (threadIdx.x == 0) {
    const float logZ   = row_max + logf(denom);
    const float xt     = x[t];
    const float mean_x = sumx / (float)N;
    const float one_hot_ce = (logZ - xt);
    const float uni_ce     = (logZ - mean_x);
    loss_vec[row] = (1.f - ls_eps) * one_hot_ce + ls_eps * uni_ce;
  }
}

template<int BS>
__global__ void ce_bwd_logits_kernel(const float* __restrict__ X,
                                     const int32_t* __restrict__ T,
                                     float* __restrict__ dX,
                                     int M, int N,
                                     float inv_scale,
                                     int ignore_index,
                                     float ls_eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  float* dx = dX + row * N;

  if (t == ignore_index) {
    for (int i = threadIdx.x; i < N; i += BS) dx[i] = 0.f;
    return;
  }

  const float* x = X + row * N;
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < N; i += BS) local_max = fmaxf(local_max, x[i]);
  float wmax = warp_max(local_max);
  __shared__ float warp_buf_max[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_buf_max[warp_id] = wmax;
  __syncthreads();

  float row_max = -CUDART_INF_F;
  if (warp_id == 0) {
    float tmax = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
    tmax = warp_max(tmax);
    if (threadIdx.x == 0) row_max = tmax;
  }
  __syncthreads();
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = row_max;
  __syncthreads();
  row_max = s_max;

  float local_e = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    float ez = __expf(x[i] - row_max);
    dx[i] = ez;  // temp
    local_e += ez;
  }
  float we = warp_sum(local_e);
  __shared__ float warp_buf_e[BS/32];
  if ((threadIdx.x & 31) == 0) warp_buf_e[warp_id] = we;
  __syncthreads();

  float denom = 0.f;
  if (warp_id == 0) {
    float te = (threadIdx.x < (BS/32)) ? warp_buf_e[threadIdx.x] : 0.f;
    te = warp_sum(te);
    if (threadIdx.x == 0) denom = te;
  }
  __syncthreads();
  __shared__ float s_den;
  if (threadIdx.x == 0) s_den = denom;
  __syncthreads();
  denom = s_den;

  const float uni = ls_eps / (float)N;
  for (int i = threadIdx.x; i < N; i += BS) {
    const float p  = dx[i] / denom;
    const float oh = (i == t) ? 1.f : 0.f;
    const float q  = (1.f - ls_eps) * oh + uni;
    dx[i] = (p - q) * inv_scale;
  }
}

template<int BS>
__global__ void ce_fwd_probs_kernel(const float* __restrict__ P,
                                    const int32_t* __restrict__ T,
                                    float* __restrict__ loss_vec,
                                    int M, int N,
                                    int ignore_index,
                                    float eps, float ls_eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  if (t == ignore_index) {
    if (threadIdx.x == 0) loss_vec[row] = 0.f;
    return;
  }

  float local_sum_logp = 0.f;

  for (int i = threadIdx.x; i < N; i += BS) {
    float p = fmaxf(P[row * N + i], eps);
    float lp = logf(p);
    local_sum_logp += lp;
    
  }
  float wsum = warp_sum(local_sum_logp);
  __shared__ float warp_buf[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_buf[warp_id] = wsum;
  __syncthreads();

  float sum_logp = 0.f;
  if (warp_id == 0) {
    float t = (threadIdx.x < (BS/32)) ? warp_buf[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) sum_logp = t;
  }
  __syncthreads();
  __shared__ float s_sum_logp, s_logp_t;
  if (threadIdx.x == 0) {
    s_sum_logp = sum_logp;
    // ★ 정확한 logp_t를 단일 스레드가 직접 읽어서 브로드캐스트
    float pt = fmaxf(P[row * N + t], eps);
    s_logp_t = logf(pt);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const float one_hot_ce = -s_logp_t;
    const float uni_ce     = -(s_sum_logp / (float)N);
    loss_vec[row] = (1.f - ls_eps) * one_hot_ce + ls_eps * uni_ce;
  }
}

template<int BS>
__global__ void ce_bwd_probs_kernel(const float* __restrict__ P,
                                    const int32_t* __restrict__ T,
                                    float* __restrict__ dX,
                                    int M, int N,
                                    float inv_scale,
                                    int ignore_index,
                                    float eps, float ls_eps)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  float* dx = dX + row * N;

  if (t == ignore_index) {
    for (int i = threadIdx.x; i < N; i += BS) dx[i] = 0.f;
    return;
  }

  const float uni = ls_eps / (float)N;
  for (int i = threadIdx.x; i < N; i += BS) {
    float p = fmaxf(P[row * N + i], eps);
    const float oh = (i == t) ? 1.f : 0.f;
    const float q  = (1.f - ls_eps) * oh + uni;
    dx[i] = (- q / p) * inv_scale;
  }
}

} // anonymous

namespace ai {

void ce_forward_logits_kernel_launcher(const float* X,
                                       const int32_t* T,
                                       float* loss_vec,
                                       int M, int N,
                                       int ignore_index,
                                       float ls_eps,
                                       cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_fwd_logits_kernel<BS><<<grid, block, 0, s>>>(X, T, loss_vec, M, N, ignore_index, ls_eps);
}

void ce_backward_logits_kernel_launcher(const float* X,
                                        const int32_t* T,
                                        float* dX,
                                        int M, int N,
                                        float inv_scale,
                                        int ignore_index,
                                        float ls_eps,
                                        cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_bwd_logits_kernel<BS><<<grid, block, 0, s>>>(X, T, dX, M, N, inv_scale, ignore_index, ls_eps);
}

void ce_forward_probs_kernel_launcher(const float* P,
                                      const int32_t* T,
                                      float* loss_vec,
                                      int M, int N,
                                      int ignore_index,
                                      float eps, float ls_eps,
                                      cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_fwd_probs_kernel<BS><<<grid, block, 0, s>>>(P, T, loss_vec, M, N, ignore_index, eps, ls_eps);
}

void ce_backward_probs_kernel_launcher(const float* P,
                                       const int32_t* T,
                                       float* dX,
                                       int M, int N,
                                       float inv_scale,
                                       int ignore_index,
                                       float eps, float ls_eps,
                                       cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_bwd_probs_kernel<BS><<<grid, block, 0, s>>>(P, T, dX, M, N, inv_scale, ignore_index, eps, ls_eps);
}

} // namespace ai
