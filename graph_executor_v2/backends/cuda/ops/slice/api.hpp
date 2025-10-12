// backends/cuda/ops/slice/api.hpp
#pragma once
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct SliceAttrs {
  // rank <= 4, step은 1만 지원 (캡처-세이프 간단버전)
  int starts[4]{0,0,0,0};
  int sizes [4]{1,1,1,1}; // 출력 크기
  int rank{1};
};

Status SliceCudaLaunch(const Tensor& X, Tensor& Y,
                       const SliceAttrs& attrs, StreamHandle stream);

} // namespace ai
