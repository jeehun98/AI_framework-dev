#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp" // Status, StreamHandle

namespace ai {

struct SoftmaxAttrs {
  float scale{1.0f};   // y = softmax(scale * x) ; scale=1/T
  bool  log{false};    // true면 logsoftmax
};

// X:[M,N] (+ optional mask:[M,N] or [1,N]) -> Y:[M,N]
Status SoftmaxCudaLaunch(const Tensor& X,
                         const Tensor* Mask,   // null 가능; 마스크는 x에 더해짐(예: -inf 또는 0)
                         Tensor& Y,
                         const SoftmaxAttrs& attrs,
                         StreamHandle stream);

// Backward: dY -> dX
// 필요시 Y를 함께 받아 재계산 방지(옵션): Y가 null이면 내부에서 한 번 더 forward 계산
Status SoftmaxCudaBackwardLaunch(const Tensor& Y_or_X, // Y (권장) 또는 X
                                 const Tensor* Mask,   // null 가능
                                 const Tensor& dY,
                                 Tensor& dX,
                                 const SoftmaxAttrs& attrs,
                                 bool y_provided,      // true면 첫 인자는 Y
                                 StreamHandle stream);

} // namespace ai
