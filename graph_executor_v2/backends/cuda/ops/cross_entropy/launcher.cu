#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "backends/cuda/ops/cross_entropy/api.hpp"

namespace ai {

// ---- 외부 선언: kernels.cu에서 제공 ----
extern void ce_forward_logits_kernel_launcher(const float*, const int32_t*, float*, int,int,int,float,cudaStream_t);
extern void ce_backward_logits_kernel_launcher(const float*, const int32_t*, float*, int,int,float,int,float,cudaStream_t);
extern void ce_forward_probs_kernel_launcher (const float*, const int32_t*, float*, int,int,int,float,float,cudaStream_t);
extern void ce_backward_probs_kernel_launcher(const float*, const int32_t*, float*, int,int,float,int,float,float,cudaStream_t);

// ---- 간단 유틸 ----
static inline cudaError_t d2d_copy(void* dst, const void* src, size_t bytes, cudaStream_t s){
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
}

// ---- n_valid 카운트 (int32 target 전제) ----
__global__ void count_valid_i32_kernel(const int32_t* T, int M, int ignore_index, int* out){
  int local = 0;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x){
    local += (T[i] != ignore_index) ? 1 : 0;
  }
  __shared__ int ssum[256];
  ssum[threadIdx.x] = local;
  __syncthreads();
  for(int o=128;o>0;o>>=1){
    if(threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
    __syncthreads();
  }
  if(threadIdx.x == 0) atomicAdd(out, ssum[0]);
}

// ---- loss[M] -> loss[1] (Sum/Mean) 리덕션 ----
template<int BS=256>
__global__ void reduce_loss_kernel(const float* __restrict__ loss_vec,
                                   const int32_t* __restrict__ T,
                                   float* __restrict__ loss_out,
                                   int M, int ignore_index,
                                   int reduction) // 1:Mean, 2:Sum
{
  float local_sum = 0.f;
  int   local_cnt = 0;
  for (int i = threadIdx.x; i < M; i += BS) {
    local_sum += loss_vec[i];
    local_cnt += (T[i] != ignore_index) ? 1 : 0;
  }
  // warp sum
  for (int o=16;o>0;o>>=1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, o);
    local_cnt += __shfl_down_sync(0xffffffff, local_cnt, o);
  }
  __shared__ float s_sum[BS/32];
  __shared__ int   s_cnt[BS/32];
  const int wid = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) { s_sum[wid] = local_sum; s_cnt[wid] = local_cnt; }
  __syncthreads();

  float total = 0.f; int vcnt = 0;
  if (wid == 0) {
    float ts = (threadIdx.x < (BS/32)) ? s_sum[threadIdx.x] : 0.f;
    int   tc = (threadIdx.x < (BS/32)) ? s_cnt[threadIdx.x] : 0;
    for (int o=16;o>0;o>>=1) { ts += __shfl_down_sync(0xffffffff, ts, o);
                               tc += __shfl_down_sync(0xffffffff, tc, o); }
    if (threadIdx.x == 0) { total = ts; vcnt = tc; }
  }
  __syncthreads();
  __shared__ float S_total; __shared__ int S_cnt;
  if (threadIdx.x == 0) { S_total = total; S_cnt = vcnt; }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (reduction == 2) { // Sum
      *loss_out = S_total;
    } else {              // Mean
      *loss_out = (S_cnt > 0) ? (S_total / (float)S_cnt) : 0.f;
    }
  }
}

} // anonymous


namespace ai {

Status CrossEntropyCudaLaunch(const Tensor& X,
                              const Tensor& target,
                              Tensor& loss,
                              const CrossEntropyAttrs& attrs,
                              StreamHandle stream)
{
  if (X.desc.dtype != DType::F32) return Status::Invalid;
  if (target.desc.dtype != DType::I32) return Status::Invalid; // int32만 지원
  if (X.desc.shape.size()!=2 || target.desc.shape.size()!=1) return Status::Invalid;
  if (target.desc.shape[0] != X.desc.shape[0]) return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);
  auto s = reinterpret_cast<cudaStream_t>(stream);

  const float* x_ptr   = X.data_ptr<const float>();
  const int32_t* t_ptr = target.data_ptr<const int32_t>();

  // 1) per-row loss 임시 버퍼
  float* loss_vec = nullptr;
  bool   reuse_out = false;
  if (attrs.reduction == Reduction::None) {
    // loss:[M]이어야 함 → 그대로 사용
    if (!(loss.desc.shape.size()==1 && loss.desc.shape[0]==M)) return Status::Invalid;
    loss_vec = loss.data_ptr<float>();
    reuse_out = true;
  } else {
    // loss:[1] → 임시 [M] 확보
    if (!(loss.desc.shape.size()==1 && loss.desc.shape[0]==1)) return Status::Invalid;
    cudaError_t err = cudaMalloc(&loss_vec, sizeof(float)*M);
    if (err != cudaSuccess) return Status::Invalid;
  }

  // 2) per-row 계산
  if (attrs.from_logits) {
    ce_forward_logits_kernel_launcher(x_ptr, t_ptr, loss_vec, M, N,
                                      attrs.ignore_index, attrs.ls_eps, s);
  } else {
    ce_forward_probs_kernel_launcher (x_ptr, t_ptr, loss_vec, M, N,
                                      attrs.ignore_index, attrs.eps, attrs.ls_eps, s);
  }

  // 3) reduction
  if (attrs.reduction == Reduction::None) {
    // 이미 loss_vec == loss.data
    return Status::Ok;
  } else {
    // Mean / Sum → loss[1]에 써주기
    constexpr int BS = 256;
    dim3 grid(1), block(BS);
    const int red = (attrs.reduction == Reduction::Mean) ? 1 : 2;
    reduce_loss_kernel<<<grid, block, 0, s>>>(loss_vec, t_ptr,
                                              loss.data_ptr<float>(), M,
                                              attrs.ignore_index, red);
    cudaError_t err0 = cudaGetLastError();
    cudaError_t err1 = cudaSuccess;
    if (!reuse_out && loss_vec) err1 = cudaFree(loss_vec);
    if (err0 != cudaSuccess || err1 != cudaSuccess) return Status::Invalid;
    return Status::Ok;
  }
}

Status CrossEntropyCudaBackwardLaunch(const Tensor& X,
                                      const Tensor& target,
                                      Tensor& dX,
                                      const CrossEntropyAttrs& attrs,
                                      StreamHandle stream)
{
  if (X.desc.dtype != DType::F32 || dX.desc.dtype != DType::F32) return Status::Invalid;
  if (target.desc.dtype != DType::I32) return Status::Invalid;
  if (X.desc.shape != dX.desc.shape) return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);
  auto s = reinterpret_cast<cudaStream_t>(stream);

  const float* x_ptr   = X.data_ptr<const float>();
  const int32_t* t_ptr = target.data_ptr<const int32_t>();
  float* dx_ptr        = dX.data_ptr<float>();

  // inv_scale (Mean만 유효 샘플 수로 나눔)
  float inv_scale = 1.f;
  if (attrs.reduction == Reduction::Mean) {
    int* d_cnt = nullptr;
    cudaError_t errA = cudaMalloc(&d_cnt, sizeof(int));
    if (errA != cudaSuccess) return Status::Invalid;
    cudaMemsetAsync(d_cnt, 0, sizeof(int), s);

    constexpr int BS = 256;
    int grid = (M + BS - 1) / BS;
    count_valid_i32_kernel<<<grid, BS, 0, s>>>(t_ptr, M, attrs.ignore_index, d_cnt);
    int h_cnt = 0;
    cudaError_t errB = cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, s);
    if (errB != cudaSuccess) { cudaFree(d_cnt); return Status::Invalid; }
    cudaError_t errS = cudaStreamSynchronize(s);
    cudaFree(d_cnt);
    if (errS != cudaSuccess) return Status::Invalid;

    inv_scale = (h_cnt > 0) ? (1.f / static_cast<float>(h_cnt)) : 0.f;
  } else {
    // None / Sum
    inv_scale = 1.f;
  }

  // backward
  if (attrs.from_logits) {
    ce_backward_logits_kernel_launcher(x_ptr, t_ptr, dx_ptr, M, N,
                                       inv_scale, attrs.ignore_index, attrs.ls_eps, s);
  } else {
    ce_backward_probs_kernel_launcher(x_ptr, t_ptr, dx_ptr, M, N,
                                      inv_scale, attrs.ignore_index, attrs.eps, attrs.ls_eps, s);
  }

  cudaError_t errK = cudaGetLastError();
  if (errK != cudaSuccess) return Status::Invalid;
  return Status::Ok;
}

} // namespace ai
