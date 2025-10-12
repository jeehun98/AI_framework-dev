// backends/cuda/ops/view/api.hpp
#pragma once
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct ViewAttrs {
  // 총 원소 수만 일치하면 OK. (RowMajor 가정 / no kernel)
  int rank{1};
  int64_t shape[4]{1,1,1,1};
};

Status ViewAliasCheck(const Tensor& X, const Tensor& Y,
                      const ViewAttrs& attrs);

} // namespace ai
