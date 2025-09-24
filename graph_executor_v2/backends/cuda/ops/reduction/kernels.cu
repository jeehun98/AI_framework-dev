#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <algorithm>
#include <math_constants.h>
#include "ai/tensor.hpp"
#include "backends/cuda/ops/reduction/api.hpp"

namespace {

__device__ __forceinline__ float op_init(ai::ReduceOp op){
  switch(op){
    case ai::ReduceOp::Sum:
    case ai::ReduceOp::Mean: return 0.f;
    case ai::ReduceOp::Max:  return -CUDART_INF_F;
    case ai::ReduceOp::Min:  return  CUDART_INF_F;
  }
  return 0.f;
}
__device__ __forceinline__ float op_apply(float acc, float v, ai::ReduceOp op){
  switch(op){
    case ai::ReduceOp::Sum:
    case ai::ReduceOp::Mean: return acc + v;
    case ai::ReduceOp::Max:  return fmaxf(acc, v);
    case ai::ReduceOp::Min:  return fminf(acc, v);
  }
  return acc;
}

// ---- 다축 일반 커널 ----
// kept(보존) 축과 reduced(축소) 축의 shape/stride를 모두 받아서 일반 인덱싱 수행
__global__ void reduce_general_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                      const int64_t* kshape,  const int64_t* kstride, int kdim,
                                      const int64_t* rshape,  const int64_t* rstride, int rdim,
                                      int64_t out_elems, ai::ReduceOp op, int64_t reduce_elems)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= out_elems) return;

  // tid -> kept 다중좌표
  int64_t base_off = 0, t = tid;
  for (int d = kdim - 1; d >= 0; --d){
    int64_t c = (kshape[d] > 0) ? (t % kshape[d]) : 0;
    t /= (kshape[d] > 0) ? kshape[d] : 1;
    base_off += c * kstride[d];
  }

  float acc = op_init(op);

  // reduce 차원 전체 누적
  for (int64_t idx = 0; idx < reduce_elems; ++idx){
    int64_t off = base_off;
    int64_t tmp = idx;
    for (int d = rdim - 1; d >= 0; --d){
      int64_t c = (rshape[d] > 0) ? (tmp % rshape[d]) : 0;
      tmp /= (rshape[d] > 0) ? rshape[d] : 1;
      off += c * rstride[d];
    }
    acc = op_apply(acc, X[off], op);
  }

  if (op == ai::ReduceOp::Mean && reduce_elems > 0){
    acc /= (float)reduce_elems;
  }
  Y[tid] = acc;
}

} // anonymous

namespace ai {

void make_rowmajor_stride(const std::vector<int64_t>& shape, std::vector<int64_t>& stride){
  const int nd = (int)shape.size();
  stride.resize(nd);
  int64_t s = 1;
  for (int i = nd-1; i >= 0; --i){ stride[i] = s; s *= shape[i]; }
}

void normalize_axes(std::vector<int>& axes, int nd){
  for (auto& a: axes) if (a < 0) a += nd;
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
}

// kept/reduced shape/stride 벡터를 받아 커널 실행
void launch_reduce_kernel(const float* X, float* Y,
                          const std::vector<int64_t>& kshape,
                          const std::vector<int64_t>& kstride,
                          const std::vector<int64_t>& rshape,
                          const std::vector<int64_t>& rstride,
                          ReduceOp op, cudaStream_t stream)
{
  const int kdim = (int)kshape.size();
  const int rdim = (int)rshape.size();

  int64_t out_elems = 1; for (auto v: kshape) out_elems *= v;
  int64_t red_elems = 1; for (auto v: rshape) red_elems *= v;

  // device copies
  int64_t *d_kshape=nullptr, *d_kstride=nullptr, *d_rshape=nullptr, *d_rstride=nullptr;
  if (kdim>0){ cudaMalloc(&d_kshape,  sizeof(int64_t)*kdim);
               cudaMalloc(&d_kstride, sizeof(int64_t)*kdim);
               cudaMemcpy(d_kshape,  kshape.data(),  sizeof(int64_t)*kdim, cudaMemcpyHostToDevice);
               cudaMemcpy(d_kstride, kstride.data(), sizeof(int64_t)*kdim, cudaMemcpyHostToDevice); }
  if (rdim>0){ cudaMalloc(&d_rshape,  sizeof(int64_t)*rdim);
               cudaMalloc(&d_rstride, sizeof(int64_t)*rdim);
               cudaMemcpy(d_rshape,  rshape.data(),  sizeof(int64_t)*rdim, cudaMemcpyHostToDevice);
               cudaMemcpy(d_rstride, rstride.data(), sizeof(int64_t)*rdim, cudaMemcpyHostToDevice); }

  const int TPB = 256;
  dim3 block(TPB);
  dim3 grid((unsigned)((out_elems + TPB - 1)/TPB));

  reduce_general_kernel<<<grid, block, 0, stream>>>(
      X, Y,
      d_kshape, d_kstride, kdim,
      d_rshape, d_rstride, rdim,
      out_elems, op, red_elems);

  if (d_kshape)  cudaFree(d_kshape);
  if (d_kstride) cudaFree(d_kstride);
  if (d_rshape)  cudaFree(d_rshape);
  if (d_rstride) cudaFree(d_rstride);
}

} // namespace ai
