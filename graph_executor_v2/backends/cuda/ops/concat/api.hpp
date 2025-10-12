// backends/cuda/ops/concat/api.hpp
#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct ConcatAttrs {
  int axis{0}; // 0..3 (rank-1도 허용)
};

Status ConcatCudaLaunch(
  const Tensor* inputs, int n_inputs, // 각 텐서: float32, row-major, rank 1~4
  Tensor& output,                     // float32, row-major
  const ConcatAttrs& attrs,
  StreamHandle stream);

} // namespace ai
