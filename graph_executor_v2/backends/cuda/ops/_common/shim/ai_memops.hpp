// backends/cuda/ops/_common/shim/ai_memops.hpp
#pragma once
#include <cuda_runtime_api.h>
#include "ai_status.hpp"
#include "ai_stream.hpp"
#include "ai_capture.hpp"

namespace ai {

// ---------------- Capture-safe memcpy/memset ----------------
inline Status ai_memcpy_async(void* dst, const void* src, size_t nbytes,
                              cudaMemcpyKind kind, StreamHandle s) {
  if (get_capture_phase(s) == CapturePhase::Invalid) return Status::RuntimeError;
  cudaError_t e = cudaMemcpyAsync(dst, src, nbytes, kind, s);
  return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
}
inline Status ai_memset_async(void* dst, int value, size_t nbytes, StreamHandle s) {
  cudaError_t e = cudaMemsetAsync(dst, value, nbytes, s);
  return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
}
inline Status memcpy_d2d_async(void* dst, const void* src, size_t nbytes, StreamHandle s) {
  return ai_memcpy_async(dst, src, nbytes, cudaMemcpyDeviceToDevice, s);
}
inline Status memcpy_h2d_async(void* dst_dev, const void* src_host, size_t nbytes, StreamHandle s) {
  return ai_memcpy_async(dst_dev, src_host, nbytes, cudaMemcpyHostToDevice, s);
}
inline Status memcpy_d2h_async(void* dst_host, const void* src_dev, size_t nbytes, StreamHandle s) {
  return ai_memcpy_async(dst_host, src_dev, nbytes, cudaMemcpyDeviceToHost, s);
}

// ---------------- Capture-safe malloc/free ----------------
inline Status ai_malloc(void** p, size_t nbytes, StreamHandle s) {
  AI_CAPTURE_FORBID_IF_ACTIVE(s, "cudaMalloc");
  cudaError_t e = cudaMalloc(p, nbytes);
  return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
}
inline Status ai_free(void* p, StreamHandle s) {
  AI_CAPTURE_FORBID_IF_ACTIVE(s, "cudaFree");
  cudaError_t e = cudaFree(p);
  return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

inline ai::Status copy_d2d_async(void* dst, const void* src, std::size_t nbytes, ai::StreamHandle stream) noexcept {
  if (nbytes == 0 || dst == src) return ai::Status::Ok;
  if (!dst || !src)              return ai::Status::Invalid;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, s);
  return (err == cudaSuccess) ? ai::Status::Ok : ai::Status::RuntimeError;
}

} // namespace ai
