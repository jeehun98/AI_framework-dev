// backends/cuda/ops/_common/shim/ai_stream.hpp
#pragma once
#include <cuda_runtime_api.h>
#include "ai_defs.hpp"  // for AI_INLINE (ensure consistent inlining)

namespace ai::cuda::shim {

// ------------------------------------------------------------
// StreamHandle: CUDA 스트림 핸들 타입 안전 래퍼
// ------------------------------------------------------------
using StreamHandle = cudaStream_t;

// 통일된 inline 규약(AI_INLINE) 적용
AI_INLINE cudaStream_t as_cuda_stream(StreamHandle s) {
  return s;
}

// Stream 상태 검증 (디버깅용)
AI_INLINE bool is_stream_valid(StreamHandle s) {
  return s != nullptr;
}

// Stream 동기화 (host-side에서만 호출)
AI_INLINE void stream_sync(StreamHandle s) {
#ifndef __CUDA_ARCH__
  cudaStreamSynchronize(as_cuda_stream(s));
#endif
}

} // namespace ai::cuda::shim
