#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

struct CrossEntropyAttrs {
  bool from_logits{true};
  Reduction reduction{Reduction::Mean};
  int ignore_index{-1};   // ✅ -1이면 비활성
  float eps{1e-9f};       // from_logits=false일 때 log(p+eps) 안정화
  float ls_eps{0.f};      // ✅ label smoothing 계수 (0=off), 추천 0.1~0.2
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
