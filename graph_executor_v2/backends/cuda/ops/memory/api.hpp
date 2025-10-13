#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"  // Tensor, Status, StreamHandle 등
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

// dst 전체에 동일 스칼라 값을 채우는 간단한 메모리 연산 (CUDA Graph 캡처 안전)
// - dst: 임의 차원, 연속(RowMajor) float32 텐서 (CUDA)
// - value: 채울 값
Status FillScalarF32CudaLaunch(Tensor& dst,
                               float value,
                               StreamHandle stream);

// int32 버전
Status FillScalarI32CudaLaunch(Tensor& dst,
                               int32_t value,
                               StreamHandle stream);

} // namespace ai
