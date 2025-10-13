#include <cuda_runtime.h>
#include "backends/cuda/ops/memory/api.hpp"

namespace ai {

static inline bool is_cuda_rowmajor_contig(const Tensor& t) {
  if (t.device != Device::CUDA) return false;
  if (t.desc.layout != Layout::RowMajor) return false;
  if (!t.data) return false;
  // 연속성: 마지막 축 stride=1, 등등 — 간단 검증
  const auto& shape = t.desc.shape;
  const auto& stride= t.desc.stride;
  if (shape.empty()) return true; // 스칼라 텐서
  if (stride.size() != shape.size()) return false;
  if (stride.back() != 1) return false;
  for (int i=(int)shape.size()-2;i>=0;--i){
    if (stride[(size_t)i] != stride[(size_t)i+1] * shape[(size_t)i+1]) return false;
  }
  return true;
}

static inline int64_t numel_of(const Tensor& t) {
  int64_t n=1;
  for (auto d: t.desc.shape) n *= d;
  return n;
}

static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// raw kernel launchers (in kernels.cu)
void fill_scalar_f32_kernel_launcher(void* dst, int64_t N, float   value, cudaStream_t s);
void fill_scalar_i32_kernel_launcher(void* dst, int64_t N, int32_t value, cudaStream_t s);

// -------- float32 --------
Status FillScalarF32CudaLaunch(Tensor& dst,
                               float value,
                               StreamHandle stream)
{
  if (!is_cuda_rowmajor_contig(dst)) return Status::Invalid;
  if (dst.desc.dtype != DType::F32)  return Status::Invalid;

  const int64_t N = numel_of(dst);
  fill_scalar_f32_kernel_launcher(dst.data, N, value, to_cuda(stream));

  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess)? Status::Ok : Status::RuntimeError;
}

// -------- int32 --------
Status FillScalarI32CudaLaunch(Tensor& dst,
                               int32_t value,
                               StreamHandle stream)
{
  if (!is_cuda_rowmajor_contig(dst)) return Status::Invalid;
  if (dst.desc.dtype != DType::I32)  return Status::Invalid;

  const int64_t N = numel_of(dst);
  fill_scalar_i32_kernel_launcher(dst.data, N, value, to_cuda(stream));

  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess)? Status::Ok : Status::RuntimeError;
}

} // namespace ai
