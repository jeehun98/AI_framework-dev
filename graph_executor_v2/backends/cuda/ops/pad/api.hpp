#pragma once
#include <vector>

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// Constant padding spec
struct PadSpec {
  std::vector<int> before;      // per-dim pads (front)
  std::vector<int> after;       // per-dim pads (back)
  float value{0.0f};            // constant fill value
};

// Forward: Y = pad(X, spec)
Status PadCudaLaunch(const Tensor& X, Tensor& Y, const PadSpec& spec, StreamHandle stream);

// Backward: dX = slice(dY, spec)  (pad 영역 제외하고 in 영역으로 복사)
Status PadBackwardCudaLaunch(const Tensor& dY, Tensor& dX, const PadSpec& spec, StreamHandle stream);

} // namespace ai
