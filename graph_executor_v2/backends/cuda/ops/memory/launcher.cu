#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstddef>

#include "backends/cuda/ops/memory/api.hpp"

// ---- Arena 헤더 (경로는 네 프로젝트에 맞게 교체) ----
// 예: #include "executor/mem_planner.hpp"
#include "src/executor/mem_planner.hpp"  // <-- 경로 조정 필요!

// kernels.cu 에서 노출한 C-링크 런처
extern "C" void fill_f32_kernel_launcher(uint64_t dst_ptr, size_t n, float value, cudaStream_t s);
extern "C" void fill_i32_kernel_launcher(uint64_t dst_ptr, size_t n, int32_t value, cudaStream_t s);

namespace ai {

// ===== 공통 유틸 =====
static inline bool is_cuda_rowmajor(const Tensor& t) {
  return t.device == Device::CUDA && t.desc.layout == Layout::RowMajor;
}
static inline bool is_f32(const Tensor& t) { return t.desc.dtype == DType::F32; }
static inline bool is_i32(const Tensor& t) { return t.desc.dtype == DType::I32; }

static inline size_t numel(const Tensor& t) {
  size_t n = 1;
  for (auto s : t.desc.shape) n *= (size_t)s;
  return n;
}
static inline cudaStream_t to_cuda(StreamHandle h) {
  return reinterpret_cast<cudaStream_t>(h);
}

// ===== Fill Ops =====
Status FillF32Cuda(Tensor& dst, float value, StreamHandle stream)
{
  if (!is_cuda_rowmajor(dst) || !is_f32(dst)) return Status::Invalid;
  const size_t n = numel(dst);
  auto s = to_cuda(stream);

  fill_f32_kernel_launcher(reinterpret_cast<uint64_t>(dst.data), n, value, s);

  // (선택) 런치 에러 체크
  // if (auto err = cudaGetLastError(); err != cudaSuccess) return Status::Invalid;
  return Status::Ok;
}

Status FillI32Cuda(Tensor& dst, int32_t value, StreamHandle stream)
{
  if (!is_cuda_rowmajor(dst) || !is_i32(dst)) return Status::Invalid;
  const size_t n = numel(dst);
  auto s = to_cuda(stream);

  fill_i32_kernel_launcher(reinterpret_cast<uint64_t>(dst.data), n, value, s);

  // if (auto err = cudaGetLastError(); err != cudaSuccess) return Status::Invalid;
  return Status::Ok;
}

// ===== Arena 어댑터 =====
//  - CaptureSafeArena는 executor 측 구현(너가 만든 mem_planner.*)
namespace arena_adaptor {
  using ::gev2::CaptureSafeArena;      // mem_planner.hpp 의 네임스페이스에 맞춰 수정
  using ::gev2::AllocStats;
  static inline cudaStream_t to_cuda(StreamHandle h) {
    return reinterpret_cast<cudaStream_t>(h);
  }
  static inline bool is_stream_capturing(cudaStream_t s) {
    cudaStreamCaptureStatus status;
    auto st = cudaStreamIsCapturing(s, &status);
    if (st == cudaErrorStreamCaptureUnsupported) return false;
    if (st != cudaSuccess) return false;
    return status != cudaStreamCaptureStatusNone;
  }
} // namespace arena_adaptor

Status MemoryReserveBytesCuda(uint64_t bytes)
{
  try {
    arena_adaptor::CaptureSafeArena::instance().reserve_bytes((size_t)bytes);
    return Status::Ok;
  } catch (...) {
    return Status::Invalid;
  }
}

Status MemoryResetPoolCuda()
{
  try {
    arena_adaptor::CaptureSafeArena::instance().reset_pool();
    return Status::Ok;
  } catch (...) {
    return Status::Invalid;
  }
}

Status MemoryStatsCuda(MemoryStats& out)
{
  try {
    auto s = arena_adaptor::CaptureSafeArena::instance().stats();
    out.total_reserved = s.total_reserved;
    out.peak_in_use    = s.peak_in_use;
    out.curr_in_use    = s.curr_in_use;
    out.slabs          = s.slabs;
    return Status::Ok;
  } catch (...) {
    return Status::Invalid;
  }
}

// 토큰 기반 temp alloc/free
Status MemoryAllocTempCuda(uint64_t nbytes,
                           uint32_t align,
                           uint64_t& out_token,
                           StreamHandle stream)
{
  try {
    auto s = arena_adaptor::to_cuda(stream);
    // CaptureSafeArena::alloc_temp 는 캡처 중 부족시 예외를 던져야 함
    uint64_t ptr = arena_adaptor::CaptureSafeArena::instance().alloc_temp(
        (size_t)nbytes, (size_t)(align ? align : 256), s);
    // ptr+size를 토큰화 (mem_planner.hpp 의 토큰 규칙과 일치해야 함)
    // 여기서는 간단히 상위 32비트에 size(4B 단위), 하위 32비트에 ptr 하위 비트
    out_token = ((uint64_t)((nbytes + 3) / 4) << 32) | (ptr & 0xffffffffull);
    return Status::Ok;
  } catch (...) {
    return Status::Invalid;
  }
}

Status MemoryFreeTempCuda(uint64_t token,
                          StreamHandle /*stream*/)
{
  try {
    // MVP: no-op(재사용 미활성) 혹은 mem_planner에서 토큰 기반 free를 구현했다면 아래로 교체
    // uint64_t ptr = (token & 0xffffffffull) | (hi_bits << 32);
    // arena_adaptor::CaptureSafeArena::instance().free_temp(ptr, arena_adaptor::to_cuda(stream));
    return Status::Ok;
  } catch (...) {
    return Status::Invalid;
  }
}

} // namespace ai
