#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

namespace { // TU-local helpers

static __device__ __forceinline__ float warp_max(float v){
  for(int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  return v;
}
static __device__ __forceinline__ float warp_sum(float v){
  for(int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// mask_kind: 0=None, 1=[M,N], 2=[1,N]/[N], 3=[M,1]
template<int BS>
__global__ void softmax_fwd_kernel(const float* __restrict__ X,
                                   const float* __restrict__ Mask, // nullable
                                   float* __restrict__ Y,
                                   int M, int N, float scale, bool logsoftmax,
                                   int mask_kind)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * N;
  float* y = Y + row * N;

  auto load_mask = [&](int i)->float{
    if (!Mask) return 0.f;
    if (mask_kind==1) return Mask[row * N + i]; // [M,N]
    if (mask_kind==2) return Mask[i];           // [1,N] or [N]
    if (mask_kind==3) return Mask[row];         // [M,1]
    return 0.f;
  };

  // 1) row-wise max
  float local_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < N; i += BS) {
    float v = x[i] + load_mask(i);
    v *= scale;
    local_max = fmaxf(local_max, v);
  }

  float wmax = warp_max(local_max);
  __shared__ float warp_buf_max[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_buf_max[warp_id] = wmax;
  __syncthreads();

  float row_max = -CUDART_INF_F;
  if (warp_id == 0) {
    float t = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
    t = warp_max(t);
    if (threadIdx.x == 0) row_max = t;
  }
  __syncthreads();
  __shared__ float s_row_max;
  if (threadIdx.x == 0) s_row_max = row_max;
  __syncthreads();
  row_max = s_row_max;

  // 2) sum(exp(z - row_max))
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    float z = (x[i] + load_mask(i)) * scale - row_max;
    float ez = __expf(z);
    y[i] = ez;  // 임시 저장
    local_sum += ez;
  }

  float wsum = warp_sum(local_sum);
  __shared__ float warp_buf_sum[BS/32];
  if ((threadIdx.x & 31) == 0) warp_buf_sum[warp_id] = wsum;
  __syncthreads();

  float denom = 0.f;
  if (warp_id == 0) {
    float t = (threadIdx.x < (BS/32)) ? warp_buf_sum[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) denom = t;
  }
  __syncthreads();
  __shared__ float s_denom;
  if (threadIdx.x == 0) s_denom = denom;
  __syncthreads();
  denom = s_denom;

  // 3) normalize or logsoftmax
  if (!logsoftmax) {
    for (int i = threadIdx.x; i < N; i += BS) y[i] = y[i] / denom;
  } else {
    float logZ = logf(denom);
    for (int i = threadIdx.x; i < N; i += BS) {
      // 현재 y[i] = exp(z - row_max) 임 → log y = (z - row_max) - logZ
      y[i] = logf(y[i]) - logZ;
    }
  }
}

template<int BS>
__global__ void softmax_bwd_kernel(const float* __restrict__ Y, // softmax or log-softmax output
                                   const float* __restrict__ dY,
                                   float* __restrict__ dX,
                                   int M, int N, float scale, bool logsoftmax)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* y  = Y  + row * N;
  const float* gy = dY + row * N;
  float* dx = dX + row * N;

  if (!logsoftmax) {
    // s = sum(dY * Y)
    float local = 0.f;
    for (int i = threadIdx.x; i < N; i += BS) local += gy[i] * y[i];
    float wsum = warp_sum(local);
    __shared__ float warp_buf[BS/32];
    const int warp_id = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) warp_buf[warp_id] = wsum;
    __syncthreads();

    float dot = 0.f;
    if (warp_id == 0) {
      float t = (threadIdx.x < (BS/32)) ? warp_buf[threadIdx.x] : 0.f;
      t = warp_sum(t);
      if (threadIdx.x == 0) dot = t;
    }
    __syncthreads();
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = dot;
    __syncthreads();
    dot = s_dot;

    for (int i = threadIdx.x; i < N; i += BS)
      dx[i] = scale * ((gy[i] - dot) * y[i]);

  } else {
    // y: log-softmax → p = exp(y)
    float local_sum_dy = 0.f;
    for (int i = threadIdx.x; i < N; i += BS) local_sum_dy += gy[i];
    float wsum = warp_sum(local_sum_dy);
    __shared__ float warp_buf[BS/32];
    const int warp_id = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) warp_buf[warp_id] = wsum;
    __syncthreads();

    float sum_dy = 0.f;
    if (warp_id == 0) {
      float t = (threadIdx.x < (BS/32)) ? warp_buf[threadIdx.x] : 0.f;
      t = warp_sum(t);
      if (threadIdx.x == 0) sum_dy = t;
    }
    __syncthreads();
    __shared__ float s_sum_dy;
    if (threadIdx.x == 0) s_sum_dy = sum_dy;
    __syncthreads();
    sum_dy = s_sum_dy;

    for (int i = threadIdx.x; i < N; i += BS) {
      float p = __expf(y[i]);               // log-softmax → softmax
      dx[i] = scale * (gy[i] - sum_dy * p);
    }
  }
}

} // anonymous

namespace ai {

void softmax_forward_kernel_launcher(const float* X, const float* Mask,
                                     float* Y, int M, int N,
                                     float scale, bool logsoftmax,
                                     int mask_kind,           // ★추가★
                                     cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  softmax_fwd_kernel<BS><<<grid, block, 0, s>>>(
      X, Mask, Y, M, N, scale, logsoftmax, mask_kind);
}

void softmax_backward_kernel_launcher(const float* Y, const float* dY,
                                      float* dX, int M, int N,
                                      float scale, bool logsoftmax, cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  softmax_bwd_kernel<BS><<<grid, block, 0, s>>>(
      Y, dY, dX, M, N, scale, logsoftmax);
}

} // namespace ai
