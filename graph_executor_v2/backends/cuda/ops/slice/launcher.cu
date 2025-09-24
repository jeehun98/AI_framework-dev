#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include "backends/cuda/ops/slice/api.hpp"

namespace {

// y_div[i] = ∏_{j>i} y_shape[j] (row-major divisor)
// x_stride[i] = ∏_{j>i} x_shape[j] (row-major stride)
__global__ void slice_kernel(const float* __restrict__ X,
                             float* __restrict__ Y,
                             int nd,
                             const int64_t* __restrict__ y_div,     // [nd]
                             const int64_t* __restrict__ x_stride,  // [nd]
                             const int64_t* __restrict__ start,     // [nd]
                             const int64_t* __restrict__ step,      // [nd]
                             int64_t y_elems)
{
  const int64_t grid_stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       tid < y_elems; tid += grid_stride)
  {
    int64_t rem  = tid;
    int64_t xoff = 0;
    #pragma unroll 4
    for (int i=0; i<nd; ++i) {
      const int64_t yi = rem / y_div[i];
      rem -= yi * y_div[i];
      const int64_t xi = start[i] + yi * step[i];
      xoff += xi * x_stride[i];
    }
    Y[tid] = X[xoff];
  }
}

static inline void make_rowmajor_stride(const std::vector<int64_t>& shape,
                                        std::vector<int64_t>& stride){
  const int nd = (int)shape.size();
  stride.assign(nd, 0);
  int64_t s=1;
  for (int i=nd-1; i>=0; --i){ stride[i]=s; s*=shape[i]; }
}

static inline int64_t numel(const std::vector<int64_t>& shape){
  int64_t n=1; for (auto v: shape) n*=v; return n;
}

static inline bool is_trivial_whole_copy(const ai::Tensor& X,
                                         const ai::Tensor& Y,
                                         const ai::SliceAttrs& a){
  const auto& xs = X.desc.shape;
  const auto& ys = Y.desc.shape;
  if (xs.size()!=ys.size()) return false;
  const int nd=(int)xs.size();
  for (int i=0;i<nd;i++){
    if (a.start[i]!=0 || a.step[i]!=1) return false;
    if (a.stop[i]!=xs[i]) return false;
    if (ys[i]!=xs[i])     return false;
  }
  return true;
}

} // anonymous

namespace ai {

Status SliceCudaLaunch(const Tensor& X, Tensor& Y,
                       const SliceAttrs& a, StreamHandle stream)
{
  const int nd = (int)X.desc.shape.size();
  if ((int)Y.desc.shape.size()!=nd) return Status::ShapeMismatch;
  if ((int)a.start.size()!=nd || (int)a.stop.size()!=nd || (int)a.step.size()!=nd)
    return Status::Invalid;

  // 기본 가드
  for (int i=0;i<nd;i++){
    if (a.step[i] <= 0)           return Status::Invalid;
    if (a.start[i] <  0)          return Status::Invalid;
    if (a.stop[i]  <  a.start[i]) return Status::Invalid;
    if (a.stop[i]  >  X.desc.shape[i]) return Status::Invalid;
  }

  const int64_t y_elems = numel(Y.desc.shape);
  if (y_elems == 0) return Status::Ok;

  auto s = (cudaStream_t)stream;

  // fast path: 전체 복사
  if (is_trivial_whole_copy(X, Y, a)){
    const size_t bytes = (size_t)y_elems * sizeof(float);
    auto err = cudaMemcpyAsync(Y.data, X.data, bytes,
                               cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) return Status::Invalid;
    err = cudaStreamSynchronize(s);
    return (err==cudaSuccess) ? Status::Ok : Status::Invalid;
  }

  // 파라미터 준비
  std::vector<int64_t> ydiv, xstride;
  make_rowmajor_stride(Y.desc.shape, ydiv);
  make_rowmajor_stride(X.desc.shape, xstride);

  // 디바이스 버퍼
  int64_t *d_ydiv=nullptr, *d_xstride=nullptr, *d_start=nullptr, *d_step=nullptr;

  auto free_all = [&](){
    if (d_ydiv)    cudaFree(d_ydiv);
    if (d_xstride) cudaFree(d_xstride);
    if (d_start)   cudaFree(d_start);
    if (d_step)    cudaFree(d_step);
    d_ydiv=d_xstride=d_start=d_step=nullptr;
  };

  if (cudaSuccess != cudaMalloc((void**)&d_ydiv,    sizeof(int64_t)*nd)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMalloc((void**)&d_xstride, sizeof(int64_t)*nd)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMalloc((void**)&d_start,   sizeof(int64_t)*nd)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMalloc((void**)&d_step,    sizeof(int64_t)*nd)) { free_all(); return Status::Invalid; }

  if (cudaSuccess != cudaMemcpyAsync(d_ydiv,    ydiv.data(),    sizeof(int64_t)*nd, cudaMemcpyHostToDevice, s)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMemcpyAsync(d_xstride, xstride.data(), sizeof(int64_t)*nd, cudaMemcpyHostToDevice, s)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMemcpyAsync(d_start,   a.start.data(), sizeof(int64_t)*nd, cudaMemcpyHostToDevice, s)) { free_all(); return Status::Invalid; }
  if (cudaSuccess != cudaMemcpyAsync(d_step,    a.step.data(),  sizeof(int64_t)*nd, cudaMemcpyHostToDevice, s)) { free_all(); return Status::Invalid; }

  // 런치 (grid-stride)
  const int BS = 256;
  int blocks = (int)std::min<int64_t>((y_elems + BS - 1) / BS, 65535);

  slice_kernel<<<blocks, BS, 0, s>>>(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      nd,
      d_ydiv,
      d_xstride,
      d_start,
      d_step,
      y_elems);

  // 에러 & 동기화 (free 전에!)
  auto kerr = cudaGetLastError();
  auto serr = cudaStreamSynchronize(s);

  free_all();

  if (kerr != cudaSuccess) return Status::RuntimeError;
  if (serr != cudaSuccess) return Status::Invalid;
  return Status::Ok;
}

} // namespace ai
