// backends/cuda/ops/_common/shim/ai_memops.hpp
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

#include "ai_defs.hpp"
#include "ai_status.hpp"
#include "ai_stream.hpp"
#include "ai_capture.hpp"
#include "ai_cuda_check.hpp"

namespace ai::cuda::shim {

// ---- (옵션) 디버그: 호스트 포인터 pinned 여부 점검 ----
#if defined(AI_SHIM_DEBUG)
AI_INLINE inline bool is_pinned_host_ptr(const void* p) noexcept {
  if (!p) return false;
  cudaPointerAttributes attr{};
  // cudaPointerGetAttributes는 버전에 따라 필드가 다름. 실패 시 false.
  if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) return false;
  // 신규 런타임에선 attr.type == cudaMemoryTypeHost, 구버전에선 isManaged/devicePointer 조합 등
  #if CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeHost;
  #else
    return attr.memoryType == cudaMemoryTypeHost;
  #endif
}
#endif

// ---- 공통 memcpy async 구현 ----
AI_INLINE inline Status memcpy_async_impl(
    void* dst, const void* src, std::size_t nbytes,
    cudaMemcpyKind kind, StreamHandle stream) noexcept
{
  if (nbytes == 0 || dst == src) return Status::Ok;
  if (!dst || !src)              return Status::Invalid;

  AI_CAPTURE_FORBID_IF_ACTIVE(stream, "cudaMemcpyAsync");

#if defined(AI_SHIM_DEBUG)
  // H2D/D2H 시 비핀 메모리 경고(동기화나 성능 저하 가능)
  if (kind == cudaMemcpyHostToDevice) {
    if (!is_pinned_host_ptr(src)) { AI_NVTX_MARK("warn: H2D non-pinned"); }
  } else if (kind == cudaMemcpyDeviceToHost) {
    if (!is_pinned_host_ptr(dst)) { AI_NVTX_MARK("warn: D2H non-pinned"); }
  }
#endif

  auto s = as_cuda_stream(stream);
  AI_CUDA_TRY(cudaMemcpyAsync(dst, src, nbytes, kind, s));
  return Status::Ok;
}

// ---- 방향별 얇은 래퍼 ----
AI_INLINE inline Status copy_d2d_async(void* dst, const void* src, std::size_t nbytes, StreamHandle stream) noexcept {
  return memcpy_async_impl(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
}
AI_INLINE inline Status copy_h2d_async(void* dst_dev, const void* src_host, std::size_t nbytes, StreamHandle stream) noexcept {
  return memcpy_async_impl(dst_dev, src_host, nbytes, cudaMemcpyHostToDevice, stream);
}
AI_INLINE inline Status copy_d2h_async(void* dst_host, const void* src_dev, std::size_t nbytes, StreamHandle stream) noexcept {
  return memcpy_async_impl(dst_host, src_dev, nbytes, cudaMemcpyDeviceToHost, stream);
}

// ---- UVA에서 kind 추론 경로(선택) ----
AI_INLINE inline Status copy_default_async(void* dst, const void* src, std::size_t nbytes, StreamHandle stream) noexcept {
  if (nbytes == 0 || dst == src) return Status::Ok;
  if (!dst || !src)              return Status::Invalid;
  AI_CAPTURE_FORBID_IF_ACTIVE(stream, "cudaMemcpyAsync(Default)");
  auto s = as_cuda_stream(stream);
  AI_CUDA_TRY(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, s));
  return Status::Ok;
}

// ---- memset: 바이트 기반. 비 0 float 초기화에는 사용 금지 ----
AI_INLINE inline Status set_d_async(void* dst, int value, std::size_t nbytes, StreamHandle stream) noexcept {
  if (nbytes == 0) return Status::Ok;
  if (!dst)        return Status::Invalid;
  AI_CAPTURE_FORBID_IF_ACTIVE(stream, "cudaMemsetAsync");
  auto s = as_cuda_stream(stream);
  AI_CUDA_TRY(cudaMemsetAsync(dst, value, nbytes, s));
  return Status::Ok;
}

// 안전한 0 초기화 별도 제공(가독성 & 오용 방지)
AI_INLINE inline Status memset_zero_d(void* dst, std::size_t nbytes, StreamHandle stream) noexcept {
  return set_d_async(dst, /*value=*/0, nbytes, stream);
}

// ---- alloc/free: 캡처 중 금지 ----
AI_INLINE inline Status alloc_d(void** p, std::size_t nbytes, StreamHandle stream) noexcept {
  if (!p) return Status::Invalid;
  AI_CAPTURE_FORBID_IF_ACTIVE(stream, "cudaMalloc");
  AI_CUDA_TRY(cudaMalloc(p, nbytes));
  return Status::Ok;
}

AI_INLINE inline Status free_d(void* p, StreamHandle stream) noexcept {
  if (!p) return Status::Ok;
  AI_CAPTURE_FORBID_IF_ACTIVE(stream, "cudaFree");
  AI_CUDA_TRY(cudaFree(p));
  return Status::Ok;
}

// ---- Deprecated aliases (1~2 릴리즈 후 제거) ----
AI_INLINE inline Status ai_memcpy_async(void* dst, const void* src, std::size_t nbytes,
                                        cudaMemcpyKind kind, StreamHandle stream) noexcept {
  return memcpy_async_impl(dst, src, nbytes, kind, stream);
}
AI_INLINE inline Status ai_memset_async(void* dst, int value, std::size_t nbytes,
                                        StreamHandle stream) noexcept {
  return set_d_async(dst, value, nbytes, stream);
}
AI_INLINE inline Status memcpy_d2d_async(void* dst, const void* src,
                                         std::size_t nbytes, StreamHandle stream) noexcept {
  return copy_d2d_async(dst, src, nbytes, stream);
}
AI_INLINE inline Status memcpy_h2d_async(void* dst_dev, const void* src_host,
                                         std::size_t nbytes, StreamHandle stream) noexcept {
  return copy_h2d_async(dst_dev, src_host, nbytes, stream);
}
AI_INLINE inline Status memcpy_d2h_async(void* dst_host, const void* src_dev,
                                         std::size_t nbytes, StreamHandle stream) noexcept {
  return copy_d2h_async(dst_host, src_dev, nbytes, stream);
}

} // namespace ai::cuda::shim
