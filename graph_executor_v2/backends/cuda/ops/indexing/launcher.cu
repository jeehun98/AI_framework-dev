#include <cuda_runtime.h>
#include "backends/cuda/ops/indexing/api.hpp"

namespace ai {

static inline bool is_f32_cuda_contig_nd(const Tensor& t){
  if (t.device != Device::CUDA || t.desc.dtype != DType::F32 || t.desc.layout != Layout::RowMajor) return false;
  // contiguous 체크
  int64_t stride = 1;
  for (int i = (int)t.desc.shape.size()-1; i >= 0; --i) {
    if (t.desc.stride[i] != stride) return false;
    stride *= t.desc.shape[i];
  }
  return true;
}
static inline bool is_i32_cuda_contig_nd(const Tensor& t){
  if (t.device != Device::CUDA || t.desc.dtype != DType::I32 || t.desc.layout != Layout::RowMajor) return false;
  int64_t stride = 1;
  for (int i = (int)t.desc.shape.size()-1; i >= 0; --i) {
    if (t.desc.stride[i] != stride) return false;
    stride *= t.desc.shape[i];
  }
  return true;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// kernels.cu 내 내부 런처
void gather_axis_launch(const float*, const int32_t*, float*, int,int,int,int, cudaStream_t);
void scatter_add_axis_launch(float*, const int32_t*, const float*, int,int,int,int, cudaStream_t);

static inline Status check_cuda_last(){
  auto e = cudaPeekAtLastError();
  return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status GatherCudaLaunch(const Tensor& X, const Tensor& Index, Tensor& Y,
                        int axis, StreamHandle stream)
{
  if (!is_f32_cuda_contig_nd(X) || !is_i32_cuda_contig_nd(Index) || !is_f32_cuda_contig_nd(Y))
    return Status::Invalid;
  int nd = (int)X.desc.shape.size();
  if ((int)Index.desc.shape.size() != nd || (int)Y.desc.shape.size() != nd)
    return Status::ShapeMismatch;
  if (axis < 0) axis += nd;
  if (axis < 0 || axis >= nd) return Status::Invalid;

  // 각 축 길이 검증: d != axis인 축들은 X/Y/Index 동일, axis에는 Index/Y 길이(M), X는 K
  for (int d=0; d<nd; ++d){
    if (d == axis) continue;
    if (X.desc.shape[d] != Index.desc.shape[d] || X.desc.shape[d] != Y.desc.shape[d])
      return Status::ShapeMismatch;
  }
  int64_t K = X.desc.shape[axis];
  int64_t M = Index.desc.shape[axis];
  if (Y.desc.shape[axis] != M) return Status::ShapeMismatch;

  // outer/inner 계산
  int64_t outer = 1, inner = 1;
  for (int d=0; d<axis; ++d) outer *= X.desc.shape[d];
  for (int d=axis+1; d<nd; ++d) inner *= X.desc.shape[d];

  gather_axis_launch(
    static_cast<const float*>(X.data),
    static_cast<const int32_t*>(Index.data),
    static_cast<float*>(Y.data),
    (int)outer, (int)K, (int)inner, (int)M,
    to_cuda(stream)
  );
  return check_cuda_last();
}

Status ScatterAddCudaLaunch(Tensor& Out, const Tensor& Index, const Tensor& Src,
                            int axis, StreamHandle stream)
{
  if (!is_f32_cuda_contig_nd(Out) || !is_i32_cuda_contig_nd(Index) || !is_f32_cuda_contig_nd(Src))
    return Status::Invalid;
  int nd = (int)Out.desc.shape.size();
  if ((int)Index.desc.shape.size() != nd || (int)Src.desc.shape.size() != nd)
    return Status::ShapeMismatch;
  if (axis < 0) axis += nd;
  if (axis < 0 || axis >= nd) return Status::Invalid;

  // d != axis는 동일, axis에서는 Out.shape[axis]=K, Index/Src.shape[axis]=M
  for (int d=0; d<nd; ++d){
    if (d == axis) continue;
    if (Out.desc.shape[d] != Index.desc.shape[d] || Out.desc.shape[d] != Src.desc.shape[d])
      return Status::ShapeMismatch;
  }
  int64_t K = Out.desc.shape[axis];
  int64_t M = Index.desc.shape[axis];
  if (Src.desc.shape[axis] != M) return Status::ShapeMismatch;

  // outer/inner
  int64_t outer = 1, inner = 1;
  for (int d=0; d<axis; ++d) outer *= Out.desc.shape[d];
  for (int d=axis+1; d<nd; ++d) inner *= Out.desc.shape[d];

  // Out은 누적 대상 — 필요 시 호출측에서 memset(0) 결정 (Embedding grad 용도면 보통 0으로)
  scatter_add_axis_launch(
    static_cast<float*>(Out.data),
    static_cast<const int32_t*>(Index.data),
    static_cast<const float*>(Src.data),
    (int)outer, (int)K, (int)inner, (int)M,
    to_cuda(stream)
  );
  return check_cuda_last();
}

} // namespace ai
