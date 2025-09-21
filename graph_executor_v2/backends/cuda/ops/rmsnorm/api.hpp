#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp" // Status, StreamHandle

namespace ai {

struct RMSNormAttrs { float eps{1e-6f}; };

// X:[M,N], gamma:[N]? beta:[N]? -> Y:[M,N]
Status RMSNormCudaLaunch(const Tensor& X,
                         const Tensor* gamma,
                         const Tensor* beta,
                         Tensor& Y,
                         const RMSNormAttrs& attrs,
                         StreamHandle stream);

// Backward: dY given -> dX[, dgamma, dbeta]
// X:[M,N], gamma:[N]? (null이면 1로 간주)
// dY:[M,N] -> dX:[M,N], dgamma:[N]? dbeta:[N]?
Status RMSNormCudaBackwardLaunch(const Tensor& X,
                                 const Tensor* gamma,
                                 const Tensor& dY,
                                 Tensor& dX,
                                 Tensor* dgamma,  // null 가능
                                 Tensor* dbeta,   // null 가능
                                 const RMSNormAttrs& attrs,
                                 StreamHandle stream);

} // namespace ai
