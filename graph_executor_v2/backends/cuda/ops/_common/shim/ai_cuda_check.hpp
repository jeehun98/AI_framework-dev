// backends/cuda/ops/_common/shim/ai_cuda_check.hpp
#pragma once
#include <cuda_runtime.h>      // cudaGetLastError 등
#include "ai_defs.hpp"
#include "ai_status.hpp"

namespace ai::cuda::shim {

AI_INLINE constexpr Status status_from_cuda(cudaError_t e) noexcept {
  return (e == cudaSuccess) ? Status::Ok : Status::CUDA_ERROR;
}

} // namespace ai::cuda::shim

// =====================================================
// CUDA 컴파일 경로 분기
//  - device(__CUDA_ARCH__ 정의됨): NO-OP 매크로 (컴파일만 통과)
//  - host  (미정의): Status 반환 매크로 (호출부에서 return Status;)
// =====================================================

#if defined(__CUDA_ARCH__)

// -------- device pass: NO-OP / 투명 통과 --------
#ifndef AI_CUDA_CHECK
  #define AI_CUDA_CHECK(expr)           (expr)
#endif
#ifndef AI_CUDA_TRY
  #define AI_CUDA_TRY(cuda_expr)        (cuda_expr)
#endif
#ifndef AI_CUDA_CHECK_LAUNCH
  #define AI_CUDA_CHECK_LAUNCH()        ((void)0)
#endif

#else  // host pass

// -------- host pass: 실패 시 Status::CUDA_ERROR 반환 --------
#ifndef AI_CUDA_CHECK
  #define AI_CUDA_CHECK(expr)                                                \
    do {                                                                     \
      cudaError_t _e__ = (expr);                                             \
      if (_e__ != cudaSuccess)                                               \
        return ::ai::cuda::shim::Status::CUDA_ERROR;                         \
    } while (0)
#endif

#ifndef AI_CUDA_TRY
  #define AI_CUDA_TRY(cuda_expr)                                             \
    do {                                                                     \
      cudaError_t _cerr__ = (cuda_expr);                                     \
      if (_cerr__ != cudaSuccess)                                            \
        return ::ai::cuda::shim::Status::CUDA_ERROR;                         \
    } while (0)
#endif

#ifndef AI_CUDA_CHECK_LAUNCH
  #define AI_CUDA_CHECK_LAUNCH()                                             \
    do {                                                                     \
      cudaError_t _e1__ = ::cudaGetLastError();                              \
      if (_e1__ != cudaSuccess)                                              \
        return ::ai::cuda::shim::Status::CUDA_ERROR;                         \
    } while (0)
#endif

#endif // __CUDA_ARCH__
