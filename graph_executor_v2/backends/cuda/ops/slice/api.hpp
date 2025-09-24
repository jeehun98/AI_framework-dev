#pragma once
#include <vector>
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

struct SliceAttrs {
  // 각 축별 [start, stop) 및 step (step>0)
  std::vector<int64_t> start, stop, step;
};

// CUDA 런처 (정의는 .cu)
Status SliceCudaLaunch(const Tensor& X, Tensor& Y,
                       const SliceAttrs& attrs, StreamHandle stream);

} // namespace ai

// 상위/바인딩이 호출할 엔트리
namespace ai { namespace ops {

int slice_run(const ai::Tensor& X, ai::Tensor& Y,
              const ai::SliceAttrs& attrs, StreamHandle s);

}} // namespace ai::ops
