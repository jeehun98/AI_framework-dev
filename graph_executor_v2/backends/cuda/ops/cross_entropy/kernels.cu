#include <cuda_runtime.h>
#include <cfloat>
#include <math_constants.h>   // <- CUDART_INF_F, CUDART_NAN_F 등 정의

#include <cmath>

namespace {

static __device__ __forceinline__ float warp_max(float v){
  for(int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  return v;
}
static __device__ __forceinline__ float warp_sum(float v){
  for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

template<int BS>
__global__ void ce_fwd_logits_kernel(const float* __restrict__ X, // [M,N]
                                     const int32_t* __restrict__ T,// [M] (class index)
                                     float* __restrict__ loss_vec, // [M] per-sample
                                     int M, int N)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * N;
  const int t = (int)T[row];

  // row max
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < N; i += BS) local_max = fmaxf(local_max, x[i]);
  float wmax = warp_max(local_max);
  __shared__ float warp_buf_max[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31)==0) warp_buf_max[warp_id] = wmax;
  __syncthreads();
  float row_max = -CUDART_INF_F;
  if (warp_id==0) {
    float tmax = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
    tmax = warp_max(tmax);
    if (threadIdx.x==0) row_max = tmax;
  }
  __syncthreads();
  __shared__ float s_max;
  if (threadIdx.x==0) s_max = row_max;
  __syncthreads();
  row_max = s_max;

  // logsumexp
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    local_sum += __expf(x[i] - row_max);
  }
  float wsum = warp_sum(local_sum);
  __shared__ float warp_buf_sum[BS/32];
  if ((threadIdx.x & 31)==0) warp_buf_sum[warp_id] = wsum;
  __syncthreads();

  float denom = 0.f;
  if (warp_id==0) {
    float tsum = (threadIdx.x < (BS/32)) ? warp_buf_sum[threadIdx.x] : 0.f;
    tsum = warp_sum(tsum);
    if (threadIdx.x==0) denom = tsum;
  }
  __syncthreads();
  __shared__ float s_den;
  if (threadIdx.x==0) s_den = denom;
  __syncthreads();
  denom = s_den;

  // loss_i = -x[t] + (row_max + log(sum exp(x-row_max)))
  float loss_i = 0.f;
  if (threadIdx.x==0) {
    float logZ = row_max + logf(denom);
    float xt  = x[(int)t];
    loss_i = (logZ - xt);
    loss_vec[row] = loss_i;
  }
}

template<int BS>
__global__ void ce_bwd_logits_kernel(const float* __restrict__ X, // [M,N]
                                     const int32_t* __restrict__ T,
                                     float* __restrict__ dX, // [M,N]
                                     int M, int N, bool mean_reduction)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * N;
  float* dx = dX + row * N;
  const int t = (int)T[row];

  // row max
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < N; i += BS) local_max = fmaxf(local_max, x[i]);
  float wmax = warp_max(local_max);
  __shared__ float warp_buf_max[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31)==0) warp_buf_max[warp_id] = wmax;
  __syncthreads();
  float row_max = -CUDART_INF_F;
  if (warp_id==0) {
    float tmax = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
    tmax = warp_max(tmax);
    if (threadIdx.x==0) row_max = tmax;
  }
  __syncthreads();
  __shared__ float s_max;
  if (threadIdx.x==0) s_max = row_max;
  __syncthreads();
  row_max = s_max;

  // sum exp & write softmax to dx temporarily
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    float ez = __expf(x[i] - row_max);
    dx[i] = ez;
    local_sum += ez;
  }
  float wsum = warp_sum(local_sum);
  __shared__ float warp_buf_sum[BS/32];
  if ((threadIdx.x & 31)==0) warp_buf_sum[warp_id] = wsum;
  __syncthreads();

  float denom = 0.f;
  if (warp_id==0) {
    float tsum = (threadIdx.x < (BS/32)) ? warp_buf_sum[threadIdx.x] : 0.f;
    tsum = warp_sum(tsum);
    if (threadIdx.x==0) denom = tsum;
  }
  __syncthreads();
  __shared__ float s_den;
  if (threadIdx.x==0) s_den = denom;
  __syncthreads();
  denom = s_den;

  // dX = softmax - one_hot(t)
  float scale = mean_reduction ? (1.f / (float)M) : 1.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    float p = dx[i] / denom;
    float oh = (i == (int)t) ? 1.f : 0.f;
    dx[i] = (p - oh) * scale;
  }
}

} // anonymous

namespace ai {

void ce_forward_logits_kernel_launcher(const float* X,
                                       const int32_t* T,
                                       float* loss_vec,
                                       int M, int N,
                                       cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_fwd_logits_kernel<BS><<<grid, block, 0, s>>>(X, T, loss_vec, M, N);
}

void ce_backward_logits_kernel_launcher(const float* X,
                                        const int32_t* T,
                                        float* dX,
                                        int M, int N,
                                        bool mean_reduction,
                                        cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_bwd_logits_kernel<BS><<<grid, block, 0, s>>>(X, T, dX, M, N, mean_reduction);
}

} // namespace ai
