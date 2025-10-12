#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cassert>

#include "backends/cuda/ops/slice/api.hpp"

namespace {

// row-major, contiguous, float32, CUDA
static inline bool is_f32_cuda_nd(const ai::Tensor& t, int max_rank=4){
  using namespace ai;
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         (int)t.desc.shape.size()<=max_rank;
}

// host에서 row-major stride 재계산 (int32)
static inline void make_rm_stride_i32(const std::vector<int64_t>& dims64, int* strides4, int rank){
  int dims4[4]={1,1,1,1};
  for (int i=0;i<rank;++i) dims4[i] = (int)dims64[i];

  int s=1;
  for (int i=rank-1;i>=0;--i){
    strides4[i] = s;
    s *= std::max(dims4[i],1);
  }
}

static inline void copy_dims_i32(const std::vector<int64_t>& dims64, int* dims4, int rank){
  for (int i=0;i<4;++i) dims4[i] = (i<rank)? (int)dims64[i] : 1;
}

} // anon

// 커널 런처 extern
extern "C" void slice_forward_kernel_launcher(const float*, float*, int,
                                              const int*, const int*, const int*, const int*, const int*, cudaStream_t);
extern "C" void slice_backward_kernel_launcher(const float*, float*, int,
                                               const int*, const int*, const int*, const int*, const int*, cudaStream_t);

namespace ai {

static inline cudaStream_t to_cuda(ai::StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

Status SliceCudaLaunch(const Tensor& X, Tensor& Y, const SliceAttrs& a, StreamHandle stream)
{
  if (!is_f32_cuda_nd(X) || !is_f32_cuda_nd(Y)) return Status::Invalid;
  const int xr = (int)X.desc.shape.size();
  const int yr = (int)Y.desc.shape.size();
  if (a.rank<1 || a.rank>4 || xr!=yr || xr!=a.rank) return Status::Invalid;

  // 경계 & shape 확인
  for (int d=0; d<a.rank; ++d){
    int xdim = (int)X.desc.shape[d];
    int ydim = (int)Y.desc.shape[d];
    int st   = a.starts[d];
    int sz   = a.sizes[d];
    if (st < 0 || sz < 1) return Status::Invalid;
    if (st + sz > xdim)   return Status::ShapeMismatch;
    if (ydim != sz)       return Status::ShapeMismatch;
  }

  // host 준비 (int32)
  int x_dims[4]={1,1,1,1}, y_dims[4]={1,1,1,1};
  int x_strides[4]={1,1,1,1}, y_strides[4]={1,1,1,1};
  int starts[4]={0,0,0,0};

  copy_dims_i32(X.desc.shape, x_dims, a.rank);
  copy_dims_i32(Y.desc.shape, y_dims, a.rank);
  make_rm_stride_i32(X.desc.shape, x_strides, a.rank); // 안전하게 재계산
  make_rm_stride_i32(Y.desc.shape, y_strides, a.rank);

  for (int i=0;i<4;++i) starts[i] = (i<a.rank)? a.starts[i] : 0;

  auto s = to_cuda(stream);

  slice_forward_kernel_launcher(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      a.rank,
      x_dims, x_strides,
      y_dims, y_strides,
      starts,
      s
  );

  return Status::Ok;
}

Status SliceCudaBackwardLaunch(const Tensor& gY, Tensor& gX, const SliceAttrs& a, StreamHandle stream)
{
  if (!is_f32_cuda_nd(gY) || !is_f32_cuda_nd(gX)) return Status::Invalid;
  const int xr = (int)gX.desc.shape.size();
  const int yr = (int)gY.desc.shape.size();
  if (a.rank<1 || a.rank>4 || xr!=yr || xr!=a.rank) return Status::Invalid;

  // 경계 & shape 확인
  for (int d=0; d<a.rank; ++d){
    int xdim = (int)gX.desc.shape[d];
    int ydim = (int)gY.desc.shape[d];
    int st   = a.starts[d];
    int sz   = a.sizes[d];
    if (st < 0 || sz < 1) return Status::Invalid;
    if (st + sz > xdim)   return Status::ShapeMismatch;
    if (ydim != sz)       return Status::ShapeMismatch;
  }

  int x_dims[4]={1,1,1,1}, y_dims[4]={1,1,1,1};
  int x_strides[4]={1,1,1,1}, y_strides[4]={1,1,1,1};
  int starts[4]={0,0,0,0};

  copy_dims_i32(gX.desc.shape, x_dims, a.rank);
  copy_dims_i32(gY.desc.shape, y_dims, a.rank);
  make_rm_stride_i32(gX.desc.shape, x_strides, a.rank);
  make_rm_stride_i32(gY.desc.shape, y_strides, a.rank);
  for (int i=0;i<4;++i) starts[i] = (i<a.rank)? a.starts[i] : 0;

  auto s = to_cuda(stream);

  slice_backward_kernel_launcher(
      static_cast<const float*>(gY.data),
      static_cast<float*>(gX.data),
      a.rank,
      x_dims, x_strides,
      y_dims, y_strides,
      starts,
      s
  );

  return Status::Ok;
}

} // namespace ai
