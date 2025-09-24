#pragma once
#include <vector>
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

struct ConcatAttrs {
  int axis{0};
};

// CUDA 런처: D2D memcpy 기반 concat (정의는 .cu)
Status ConcatCudaLaunch(const std::vector<Tensor>& Xs,
                        Tensor& Y,
                        const ConcatAttrs& attrs,
                        StreamHandle stream);

} // namespace ai

// 상위/바인딩이 호출할 엔트리 (slice_run과 동일 패턴)
namespace ai { namespace ops {

int concat_run(const std::vector<ai::Tensor>& Xs,
               ai::Tensor& Y,
               const ai::ConcatAttrs& attrs,
               StreamHandle s);

}} // namespace ai::ops
