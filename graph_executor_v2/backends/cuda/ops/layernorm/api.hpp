// backends/cuda/ops/layernorm/api.hpp
#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp" // Status, StreamHandle

namespace ai {

struct LayerNormAttrs { float eps{1e-5f}; };

// X:[M,N], gamma:[N]? beta:[N]? -> Y:[M,N]
Status LayerNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,
                           const Tensor* beta,
                           Tensor& Y,
                           const LayerNormAttrs& attrs,
                           StreamHandle stream);

// Backward: dY -> dX[, dgamma, dbeta]
// gamma/beta가 null이면 scale/bias 없는 LN으로 처리
Status LayerNormCudaBackwardLaunch(const Tensor& X,
                                   const Tensor* gamma,       // null 가능
                                   const Tensor& dY,
                                   Tensor& dX,
                                   Tensor* dgamma,            // null 가능
                                   Tensor* dbeta,             // null 가능
                                   const LayerNormAttrs& attrs,
                                   StreamHandle stream);

} // namespace ai
