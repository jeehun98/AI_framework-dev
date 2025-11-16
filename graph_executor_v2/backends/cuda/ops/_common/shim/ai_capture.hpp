// backends/cuda/ops/_common/shim/ai_capture.hpp
#pragma once
#include <cuda_runtime_api.h>
#include "ai_status.hpp"
#include "ai_stream.hpp"   // StreamHandle, as_cuda_stream
#include "ai_defs.hpp"     // AI_INLINE

namespace ai::cuda::shim {

enum class CapturePhase : int { None = 0, Active = 1, Invalid = 2 };

AI_INLINE inline CapturePhase get_capture_phase(StreamHandle s) noexcept {
  cudaStream_t cs = as_cuda_stream(s);
  cudaStreamCaptureStatus st;
  unsigned long long id = 0;
  // graphs/nodes는 필요 없으므로 nullptr 전달
  cudaError_t e = cudaStreamGetCaptureInfo_v2(cs, &st, &id, nullptr, nullptr);
  if (e != cudaSuccess)                 return CapturePhase::Invalid;
  if (st == cudaStreamCaptureStatusActive) return CapturePhase::Active;
  if (st == cudaStreamCaptureStatusNone)   return CapturePhase::None;
  return CapturePhase::Invalid;
}

// 캡처 중 금지 매크로: 활성 캡처면 즉시 Status::RuntimeError 반환
#ifndef AI_CAPTURE_FORBID_IF_ACTIVE
#define AI_CAPTURE_FORBID_IF_ACTIVE(stream_handle, what)                                   \
  do {                                                                                     \
    ::ai::cuda::shim::CapturePhase _ph__ = ::ai::cuda::shim::get_capture_phase(stream_handle); \
    if (_ph__ == ::ai::cuda::shim::CapturePhase::Active) { (void)(what);                   \
      return ::ai::cuda::shim::Status::RuntimeError;                                       \
    }                                                                                      \
  } while (0)
#endif

// (선택) 캡처 상태 헬퍼
#ifndef AI_CAPTURE_IS_ACTIVE
#define AI_CAPTURE_IS_ACTIVE(stream_handle)                                                \
  (::ai::cuda::shim::get_capture_phase(stream_handle) == ::ai::cuda::shim::CapturePhase::Active)
#endif

} // namespace ai::cuda::shim
