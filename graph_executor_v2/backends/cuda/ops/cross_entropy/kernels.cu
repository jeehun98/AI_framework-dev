#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>
#include <cmath>

namespace {

// ===== warp helpers =====
__device__ __forceinline__ float warp_max(float v){
  for (int o=16;o>0;o>>=1) 
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v){
  for (int o=16;o>0;o>>=1) 
    v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// ============================ Forward (from logits) ============================
template<int BS>
__global__ void ce_fwd_logits_kernel(const float* __restrict__ X, // [M,N]
                                     const int32_t* __restrict__ T,// [M]
                                     float* __restrict__ loss_out, // [M] (reduction=None) or [1] (Mean/Sum)
                                     int M, int N,
                                     int ignore_index,
                                     float ls_eps,
                                     int reduction_kind)           // 0=None, 1=Mean, 2=Sum
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  if (t == ignore_index) {
    if (reduction_kind == 0) {
      if (threadIdx.x == 0) loss_out[row] = 0.f;
    }
    // Mean/Sum에서는 무시 샘플의 기여 0 → atomicAdd(0) 생략
    return;
  }

  const float* x = X + row * N;

  // 1) row max
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

  // 2) denom = sum(exp(x - max)), sumx = sum(x)
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

  // 3) row loss
  float loss_row = 0.f;
  if (threadIdx.x == 0) {
    const float logZ   = row_max + logf(denom);
    const float xt     = x[t];
    const float mean_x = sumx / (float)N;
    const float one_hot_ce = (logZ - xt);
    const float uni_ce     = (logZ - mean_x);
    loss_row = (1.f - ls_eps) * one_hot_ce + ls_eps * uni_ce;
  }

  // 4) write
  if (reduction_kind == 0) {
    // None: per-sample vector
    if (threadIdx.x == 0) loss_out[row] = loss_row;
  } else {
    // Mean/Sum: atomicAdd to scalar
    float scale = (reduction_kind == 1) ? (1.f / (float)M) : 1.f;
    if (threadIdx.x == 0) atomicAdd(loss_out, loss_row * scale);
  }
}

// ============================ Backward (from logits) ============================
template<int BS>
__global__ void ce_bwd_logits_kernel(const float* __restrict__ X,
                                     const int32_t* __restrict__ T,
                                     float* __restrict__ dX,
                                     int M, int N,
                                     float inv_scale,   // None/Sum=1, Mean=1/M
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

  // softmax(x)
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
    dx[i] = ez;  // temp store
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

// ============================ Forward (from probs) ============================
template<int BS>
__global__ void ce_fwd_probs_kernel(const float* __restrict__ P, // [M,N]
                                    const int32_t* __restrict__ T,
                                    float* __restrict__ loss_out, // [M] or [1]
                                    int M, int N,
                                    int ignore_index,
                                    float eps, float ls_eps,
                                    int reduction_kind)           // 0=None,1=Mean,2=Sum
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int t = static_cast<int>(T[row]);
  if (t == ignore_index) {
    if (reduction_kind == 0) {
      if (threadIdx.x == 0) loss_out[row] = 0.f;
    }
    return;
  }

  // sum log p, and log p_t
  float local_sum_logp = 0.f;
  for (int i = threadIdx.x; i < N; i += BS) {
    float p = fmaxf(P[row * N + i], eps);
    local_sum_logp += logf(p);
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
    float pt = fmaxf(P[row * N + t], eps);
    s_logp_t = logf(pt);
  }
  __syncthreads();

  float loss_row = 0.f;
  if (threadIdx.x == 0) {
    const float one_hot_ce = -s_logp_t;
    const float uni_ce     = -(s_sum_logp / (float)N);
    loss_row = (1.f - ls_eps) * one_hot_ce + ls_eps * uni_ce;
  }

  if (reduction_kind == 0) {
    if (threadIdx.x == 0) loss_out[row] = loss_row;
  } else {
    float scale = (reduction_kind == 1) ? (1.f / (float)M) : 1.f;
    if (threadIdx.x == 0) atomicAdd(loss_out, loss_row * scale);
  }
}

// ============================ Backward (from probs) ============================
template<int BS>
__global__ void ce_bwd_probs_kernel(const float* __restrict__ P,
                                    const int32_t* __restrict__ T,
                                    float* __restrict__ dX,
                                    int M, int N,
                                    float inv_scale,  // None/Sum=1, Mean=1/M
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

// ============================ Fused Softmax + CE (from logits) ============================
//  - 한 번에: 안정 softmax → loss(write) → dlogits(write)
//  - ignore_index / label_smoothing은 단순 버전(미지원)로 시작: q = one-hot
template<int BS>
__global__ void softmax_ce_fused_fwd_bwd_kernel(const float* __restrict__ X,   // [M,C] logits
                                                const int32_t* __restrict__ T, // [M] labels
                                                float* __restrict__ dL,        // [M,C] output
                                                float* __restrict__ loss_out,  // nullptr or [M]/[1]
                                                int M, int C,
                                                int reduction_kind,            // 0=None, 1=Mean, 2=Sum
                                                int stable)                    // 0/1
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const float* x = X + row * C;
  float* dl = dL + row * C;
  const int t = static_cast<int>(T[row]);

  // softmax 안정화: stable==1이면 row_max 사용, 아니면 0으로 처리(naive)
  float row_max = 0.f;
  if (stable){
    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < C; i += BS) local_max = fmaxf(local_max, x[i]);
    float wmax = warp_max(local_max);
    __shared__ float warp_buf_max[BS/32];
    const int warp_id = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) warp_buf_max[warp_id] = wmax;
    __syncthreads();
    if (warp_id == 0){
      float tmax = (threadIdx.x < (BS/32)) ? warp_buf_max[threadIdx.x] : -CUDART_INF_F;
      tmax = warp_max(tmax);
      if (threadIdx.x == 0) row_max = tmax;
    }
    __syncthreads();
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;
  } else {
    row_max = 0.f; // naive
  }

  // denom 및 exp 저장(임시로 dL에 p분자(e^z) 저장)
  float local_e = 0.f;
  for (int i = threadIdx.x; i < C; i += BS) {
    float ez = __expf(x[i] - row_max);
    dl[i] = ez;
    local_e += ez;
  }
  float we = warp_sum(local_e);
  __shared__ float warp_buf_e[BS/32];
  const int warp_id = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_buf_e[warp_id] = we;
  __syncthreads();

  float denom = 0.f;
  if (warp_id == 0){
    float te = (threadIdx.x < (BS/32)) ? warp_buf_e[threadIdx.x] : 0.f;
    te = warp_sum(te);
    if (threadIdx.x == 0) denom = te;
  }
  __syncthreads();
  __shared__ float s_den;
  if (threadIdx.x == 0) s_den = denom;
  __syncthreads();
  denom = s_den;

  // loss: logZ - x_t  (Z=∑exp, logZ = row_max + log(denom))  — one-hot 기준
  float loss_row = 0.f;
  if (threadIdx.x == 0 && loss_out){
    const float logZ = (stable ? row_max : 0.f) + logf(denom);
    const float xt   = x[t];
    loss_row = (logZ - xt);
    if (reduction_kind == 0) {
      loss_out[row] = loss_row;
    } else {
      float scale = (reduction_kind == 1) ? (1.f / (float)M) : 1.f;
      atomicAdd(loss_out, loss_row * scale);
    }
  }

  // dlogits = p - one_hot(t)
  for (int i = threadIdx.x; i < C; i += BS){
    float p = dl[i] / denom;
    float oh = (i == t) ? 1.f : 0.f;
    dl[i] = (p - oh);
  }
}

} // anonymous


// ============================ Launchers (for .cpp) ============================
namespace ai {

void ce_forward_logits_kernel_launcher(const float* X,
                                       const int32_t* T,
                                       float* loss_out,
                                       int M, int N,
                                       int ignore_index,
                                       float ls_eps,
                                       int reduction_kind,
                                       cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_fwd_logits_kernel<BS><<<grid, block, 0, s>>>(
      X, T, loss_out, M, N, ignore_index, ls_eps, reduction_kind);
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
  ce_bwd_logits_kernel<BS><<<grid, block, 0, s>>>(
      X, T, dX, M, N, inv_scale, ignore_index, ls_eps);
}

void ce_forward_probs_kernel_launcher(const float* P,
                                      const int32_t* T,
                                      float* loss_out,
                                      int M, int N,
                                      int ignore_index,
                                      float eps, float ls_eps,
                                      int reduction_kind,
                                      cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  ce_fwd_probs_kernel<BS><<<grid, block, 0, s>>>(
      P, T, loss_out, M, N, ignore_index, eps, ls_eps, reduction_kind);
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
  ce_bwd_probs_kernel<BS><<<grid, block, 0, s>>>(
      P, T, dX, M, N, inv_scale, ignore_index, eps, ls_eps);
}

// --------- NEW: Fused Softmax + CE (FWD+ BWD from logits) ----------
void softmax_ce_fused_fwd_bwd_kernel_launcher(
    const float* logits,      // [M,C]
    const int32_t* labels,    // [M]
    float* dlogits,           // [M,C]
    float* loss_out,          // nullable; [M] if red=None, [1] if red=Mean/Sum
    int M, int C,
    int reduction,            // 0:None, 1:Mean, 2:Sum
    int stable,               // 0/1
    cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  softmax_ce_fused_fwd_bwd_kernel<BS><<<grid, block, 0, s>>>(
      logits, labels, dlogits, loss_out, M, C, reduction, stable);
}

} // namespace ai
