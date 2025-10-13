#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// rank<=4 고정, I64 미지원 버전: attrs는 int32만 사용
struct SliceAttrs {
  int rank{1};
  int starts[4]{0,0,0,0};
  int sizes [4]{1,1,1,1};
};

// Forward: Y = X[starts : starts+sizes]  (row-major contiguous, float32, CUDA)
Status SliceCudaLaunch(const Tensor& X,
                       Tensor& Y,
                       const SliceAttrs& attrs,
                       StreamHandle stream);

// Backward: gX += scatter(gY)  (accumulate-add; slice 특성상 1:1 mapping, atomic 불필요)
Status SliceCudaBackwardLaunch(const Tensor& gY,
                               Tensor& gX,
                               const SliceAttrs& attrs,
                               StreamHandle stream);

} // namespace ai
