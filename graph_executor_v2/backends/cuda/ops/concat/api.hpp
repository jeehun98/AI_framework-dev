#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// I64 미지원: attrs는 int32
struct ConcatAttrs {
  int rank{1};
  int axis{0};  // 0..rank-1
};

// Forward: Y = concat(Xs, axis)
//   - Xs: 길이 n의 텐서 배열, 모두 float32 CUDA row-major, rank 동일, axis 제외 dims 동일
Status ConcatCudaLaunch(const Tensor* Xs, int n,
                        Tensor& Y,
                        const ConcatAttrs& attrs,
                        StreamHandle stream);

// Backward: 각 gX_i += slice(gY, axis, offset_i, size_i)
Status ConcatCudaBackwardLaunch(const Tensor& gY,
                                Tensor* gXs, int n,
                                const ConcatAttrs& attrs,
                                StreamHandle stream);

} // namespace ai
