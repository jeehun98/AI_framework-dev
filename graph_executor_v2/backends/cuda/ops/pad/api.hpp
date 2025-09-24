#pragma once
#include <vector>
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"  // for Status, StreamHandle

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
