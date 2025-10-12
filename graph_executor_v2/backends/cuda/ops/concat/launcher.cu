#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "backends/cuda/ops/concat/api.hpp"

namespace {

static inline bool is_f32_cuda_nd(const ai::Tensor& t, int max_rank=4){
  using namespace ai;
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         (int)t.desc.shape.size()<=max_rank;
}

static inline void make_rm_stride_i32(const std::vector<int64_t>& dims64, int* strides4, int rank){
  int dims4[4]={1,1,1,1};
  for (int i=0;i<rank;++i) dims4[i] = (int)dims64[i];
  int s=1;
  for (int d=rank-1; d>=0; --d){ strides4[d]=s; s*=std::max(dims4[d],1); }
}
static inline void dims_i64_to_i32(const std::vector<int64_t>& dims64, int* dims4, int rank){
  for (int i=0;i<4;++i) dims4[i] = (i<rank)? (int)dims64[i] : 1;
}

} // anon

extern "C" void concat_copy_region_kernel_launcher(const float*, float*, int,
  const int*, const int*, const int*, const int*, const int*, cudaStream_t);
extern "C" void concat_add_region_kernel_launcher(const float*, float*, int,
  const int*, const int*, const int*, const int*, const int*, cudaStream_t);

namespace ai {

static inline cudaStream_t to_cuda(ai::StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

Status ConcatCudaLaunch(const Tensor* Xs, int n, Tensor& Y, const ConcatAttrs& a, StreamHandle stream)
{
  if (!Xs || n<=0) return Status::Invalid;
  if (!is_f32_cuda_nd(Y)) return Status::Invalid;

  const int rank = a.rank;
  if (rank<1 || rank>4) return Status::Invalid;
  if (a.axis<0 || a.axis>=rank) return Status::Invalid;

  // 공통 차원 검사 + axis 길이 합이 Y의 axis 길이와 동일한지 확인
  std::vector<int> x_axis_sizes(n,0);
  for (int i=0;i<n;++i){
    if (!is_f32_cuda_nd(Xs[i])) return Status::Invalid;
    if ((int)Xs[i].desc.shape.size()!=rank) return Status::Invalid;
    for (int d=0; d<rank; ++d){
      int xdim = (int)Xs[i].desc.shape[d];
      int ydim = (int)Y.desc.shape[d];
      if (d==a.axis){
        x_axis_sizes[i] = xdim;
      } else {
        if (xdim != ydim) return Status::ShapeMismatch;
      }
    }
  }
  int sum_axis = 0;
  for (int v: x_axis_sizes) sum_axis += v;
  if (sum_axis != (int)Y.desc.shape[a.axis]) return Status::ShapeMismatch;

  // Y strides / dims
  int y_dims[4]={1,1,1,1}, y_strides[4]={1,1,1,1};
  dims_i64_to_i32(Y.desc.shape, y_dims, rank);
  make_rm_stride_i32(Y.desc.shape, y_strides, rank);

  auto s = to_cuda(stream);

  // 각 입력을 Y의 누적 위치에 복사
  int offset = 0;
  for (int i=0;i<n;++i){
    // reg_dims = Xi dims (전체 복사)
    int reg_dims[4]={1,1,1,1};
    int x_strides[4]={1,1,1,1};
    int x_starts[4]={0,0,0,0}, y_starts[4]={0,0,0,0};
    dims_i64_to_i32(Xs[i].desc.shape, reg_dims, rank);
    make_rm_stride_i32(Xs[i].desc.shape, x_strides, rank);
    y_starts[a.axis] = offset;

    concat_copy_region_kernel_launcher(
      static_cast<const float*>(Xs[i].data),
      static_cast<float*>(Y.data),
      rank,
      reg_dims,
      x_strides, y_strides,
      x_starts,  y_starts,
      s
    );
    offset += reg_dims[a.axis];
  }

  return Status::Ok;
}

Status ConcatCudaBackwardLaunch(const Tensor& gY, Tensor* gXs, int n, const ConcatAttrs& a, StreamHandle stream)
{
  if (!is_f32_cuda_nd(gY)) return Status::Invalid;
  if (!gXs || n<=0) return Status::Invalid;

  const int rank = a.rank;
  if (rank<1 || rank>4) return Status::Invalid;
  if (a.axis<0 || a.axis>=rank) return Status::Invalid;

  // gX shape 검사, axis 합 검증
  int sum_axis=0;
  for (int i=0;i<n;++i){
    if (!is_f32_cuda_nd(gXs[i])) return Status::Invalid;
    if ((int)gXs[i].desc.shape.size()!=rank) return Status::Invalid;
    for (int d=0; d<rank; ++d){
      int xdim = (int)gXs[i].desc.shape[d];
      int ydim = (int)gY.desc.shape[d];
      if (d==a.axis){
        sum_axis += xdim;
      } else {
        if (xdim != ydim) return Status::ShapeMismatch;
      }
    }
  }
  if (sum_axis != (int)gY.desc.shape[a.axis]) return Status::ShapeMismatch;

  int y_dims[4]={1,1,1,1}, y_strides[4]={1,1,1,1};
  dims_i64_to_i32(gY.desc.shape, y_dims, rank);
  make_rm_stride_i32(gY.desc.shape, y_strides, rank);

  auto s = to_cuda(stream);

  int offset=0;
  for (int i=0;i<n;++i){
    int reg_dims[4]={1,1,1,1};
    int d_strides[4]={1,1,1,1}; // dst: gX[i]
    int s_starts[4]={0,0,0,0}, d_starts[4]={0,0,0,0};

    dims_i64_to_i32(gXs[i].desc.shape, reg_dims, rank);
    make_rm_stride_i32(gXs[i].desc.shape, d_strides, rank);

    s_starts[a.axis] = offset; // gY region 시작
    // d_starts는 0 (각 gX는 자기 텐서의 (0..reg_dims-1))
    concat_add_region_kernel_launcher(
      static_cast<const float*>(gY.data),
      static_cast<float*>(gXs[i].data),
      rank,
      reg_dims,
      y_strides, d_strides,
      s_starts,  d_starts,
      s
    );
    offset += reg_dims[a.axis];
  }

  return Status::Ok;
}

} // namespace ai
