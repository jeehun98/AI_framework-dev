// backends/cuda/ops/_common/shim/ai_cuda_check.hpp
#pragma once
#include <cuda_runtime_api.h>
#include "ai_status.hpp"

namespace ai {} // for namespace consistency

#ifndef AI_CUDA_CHECK
#define AI_CUDA_CHECK(expr) do {                                   \
  cudaError_t _e = (expr);                                         \
  if (_e != cudaSuccess) return ::ai::Status::RuntimeError;        \
} while(0)
#endif

#ifndef AI_CUDA_TRY
#define AI_CUDA_TRY(cuda_expr) do {                                \
  cudaError_t _cerr__ = (cuda_expr);                               \
  if (_cerr__ != cudaSuccess) return ::ai::Status::RuntimeError;   \
} while(0)
#endif

#ifndef AI_CUDA_CHECK_LAUNCH
#define AI_CUDA_CHECK_LAUNCH() do {                                \
  cudaError_t _e1 = cudaGetLastError();                            \
  if (_e1 != cudaSuccess) return ::ai::Status::RuntimeError;       \
} while(0)
#endif
