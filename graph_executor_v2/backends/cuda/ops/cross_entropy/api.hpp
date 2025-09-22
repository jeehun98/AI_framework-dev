#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

struct CrossEntropyAttrs {
  bool from_logits{true};      // true: X는 logits, false: X는 확률(p)
  Reduction reduction{Reduction::Mean};
  int ignore_index{-1};        // (MVP에선 미사용; 후속 확장 시 적용)
  float eps{1e-9f};            // from_logits=false일 때 log( p + eps ) 안정화
};

// X:[M,N], target:[M] (int32/int64)  -> loss:
//   reduction=None  => loss:[M] (per-sample)
//   reduction=Mean/Sum => loss:[1] (scalar)
Status CrossEntropyCudaLaunch(const Tensor& X,
                              const Tensor& target,
                              Tensor& loss,
                              const CrossEntropyAttrs& attrs,
                              StreamHandle stream);

// dX:[M,N] (same shape as X)
Status CrossEntropyCudaBackwardLaunch(const Tensor& X,
                                      const Tensor& target,
                                      Tensor& dX,
                                      const CrossEntropyAttrs& attrs,
                                      StreamHandle stream);

} // namespace ai
