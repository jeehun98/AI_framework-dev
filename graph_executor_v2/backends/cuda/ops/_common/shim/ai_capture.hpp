// backends/cuda/ops/_common/shim/ai_capture.hpp
#pragma once
#include <cuda_runtime_api.h>
#include "ai_status.hpp"
#include "ai_stream.hpp"

namespace ai {

enum class CapturePhase { None, Active, Invalid };

inline CapturePhase get_capture_phase(StreamHandle s) {
  cudaStreamCaptureStatus st; unsigned long long id = 0;
  cudaError_t e = cudaStreamGetCaptureInfo_v2(s, &st, &id, nullptr, nullptr);
  if (e != cudaSuccess) return CapturePhase::Invalid;
  if (st == cudaStreamCaptureStatusActive) return CapturePhase::Active;
  if (st == cudaStreamCaptureStatusNone)   return CapturePhase::None;
  return CapturePhase::Invalid;
}

// 캡처 중 금지 매크로
#ifndef AI_CAPTURE_FORBID_IF_ACTIVE
#define AI_CAPTURE_FORBID_IF_ACTIVE(stream_handle, what)                             \
  do {                                                                               \
    ::ai::CapturePhase _ph = ::ai::get_capture_phase(stream_handle);                 \
    if (_ph == ::ai::CapturePhase::Active) { (void)(what);                           \
      return ::ai::Status::RuntimeError;                                             \
    }                                                                                \
  } while(0)
#endif

} // namespace ai
