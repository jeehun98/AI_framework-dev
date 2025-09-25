#include <cuda_runtime.h>
#include "backends/cuda/ops/memory/api.hpp"

namespace ai {

static inline bool is_cuda_f32(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// 커널 선언
void contiguous_copy_kernel_launcher(
    const float* src, float* dst,
    const int64_t* shape_h, const int64_t* stride_h,
    int nd, int64_t total, cudaStream_t stream);

Status ContiguousCopyCudaLaunch(const Tensor& src, Tensor& dst, StreamHandle stream)
{
  if (!is_cuda_f32(src) || !is_cuda_f32(dst)) return Status::Invalid;
  if (src.desc.shape != dst.desc.shape)       return Status::ShapeMismatch;
  if (src.desc.shape.empty())                 return Status::Invalid;

  const int nd = (int)src.desc.shape.size();
  if (nd > 8) return Status::Invalid; // MEM_MAX_NDIMS 초과 방지

  // 총 요소 수
  int64_t total = 1;
  for (auto s : src.desc.shape) total *= s;

  // host 배열 준비
  int64_t shape_h[8], stride_h[8];
  for (int i = 0; i < nd; ++i) {
    shape_h[i]  = src.desc.shape[i];
    stride_h[i] = src.desc.stride[i]; // 요소 단위 stride (프레임워크 일관성 가정)
  }

  contiguous_copy_kernel_launcher(
    static_cast<const float*>(src.data),
    static_cast<float*>(dst.data),
    shape_h, stride_h,
    nd, total, to_cuda(stream)
  );

  // 커널 런치 에러 확인
  auto e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

} // namespace ai
