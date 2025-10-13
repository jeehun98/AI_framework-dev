#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// 단순 view(reshape/alias) 검사용, I64 미지원: int32 shape
struct ViewAttrs {
  int rank{1};
  int shape[4]{1,1,1,1}; // 기대 shape (옵션)
};

// Forward: 보통 no-op. (필요시 검증만 수행)
Status ViewCudaLaunch(const Tensor& X,
                      Tensor& Y,
                      const ViewAttrs& attrs,
                      StreamHandle stream);

// Backward: gX += gY  (alias일 때 누적)
Status ViewCudaBackwardLaunch(const Tensor& gY,
                              Tensor& gX,
                              const ViewAttrs& attrs,
                              StreamHandle stream);

// 선택: alias 체크 유틸 (원하면 유지)
Status ViewAliasCheck(const Tensor& X,
                      const Tensor& Y,
                      const ViewAttrs& attrs);

} // namespace ai
