#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "backends/cuda/ops/view/api.hpp"

extern "C" void view_copy_kernel_launcher(const float*, float*, int64_t, cudaStream_t);
extern "C" void view_add_kernel_launcher (const float*, float*, int64_t, cudaStream_t);

namespace ai {

static inline bool is_f32_cuda_nd(const Tensor& t, int max_rank=4){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         (int)t.desc.shape.size()<=max_rank;
}

static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

Status ViewAliasCheck(const Tensor& X, const Tensor& Y, const ViewAttrs& a){
  const int xr = (int)X.desc.shape.size();
  const int yr = (int)Y.desc.shape.size();
  if (xr!=yr || xr<1 || xr>4) return Status::Invalid;

  // 총 원소 수 동일성 검사
  int64_t nx=1, ny=1;
  for (int d=0; d<xr; ++d){ nx *= X.desc.shape[d]; ny *= Y.desc.shape[d]; }
  if (nx!=ny) return Status::ShapeMismatch;

  // attrs.shape가 지정되었다면 Y.shape와도 맞는지 확인(옵션)
  if (a.rank==yr){
    for (int d=0; d<yr; ++d){
      if (a.shape[d]>0 && a.shape[d] != (int)Y.desc.shape[d]) return Status::ShapeMismatch;
    }
  }
  return Status::Ok;
}

Status ViewCudaLaunch(const Tensor& X, Tensor& Y, const ViewAttrs& a, StreamHandle stream)
{
  if (!is_f32_cuda_nd(X) || !is_f32_cuda_nd(Y)) return Status::Invalid;
  Status st = ViewAliasCheck(X, Y, a);
  if (st != Status::Ok) return st;

  // 보통 alias(no-copy). 포인터가 다르면 안전 복사(선택).
  if (X.data != Y.data){
    int64_t total = 1;
    for (auto v: Y.desc.shape) total *= v;
    view_copy_kernel_launcher(static_cast<const float*>(X.data),
                              static_cast<float*>(Y.data),
                              total, to_cuda(stream));
  }
  return Status::Ok;
}

Status ViewCudaBackwardLaunch(const Tensor& gY, Tensor& gX, const ViewAttrs& a, StreamHandle stream)
{
  if (!is_f32_cuda_nd(gY) || !is_f32_cuda_nd(gX)) return Status::Invalid;
  // 동일 원소 수 확인
  int64_t ny=1, nx=1;
  for (auto v: gY.desc.shape) ny *= v;
  for (auto v: gX.desc.shape) nx *= v;
  if (nx!=ny) return Status::ShapeMismatch;

  // gX += gY
  view_add_kernel_launcher(static_cast<const float*>(gY.data),
                           static_cast<float*>(gX.data),
                           ny, to_cuda(stream));
  return Status::Ok;
}

} // namespace ai
