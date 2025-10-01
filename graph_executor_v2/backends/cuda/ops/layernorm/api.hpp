// backends/cuda/ops/layernorm/api.hpp
#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

/**
 * LayerNorm: row-wise 정규화
 * - 입력 X: [M, N] (row-major, 연속 권장)
 * - gamma:  [N] or null
 * - beta :  [N] or null
 * - 출력 Y: [M, N]
 *
 * y = (x - mean) / sqrt(var + eps) * (gamma?) + (beta?)
 * (gamma/beta가 null이면 scale/bias 없는 LN)
 */
struct LayerNormAttrs { float eps{1e-5f}; };

Status LayerNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,   // null 가능
                           const Tensor* beta,    // null 가능
                           Tensor& Y,
                           const LayerNormAttrs& attrs,
                           StreamHandle stream);

/**
 * Backward:
 *  - 입력:  X:[M,N], dY:[M,N], (gamma:[N]? for dX 경로)
 *  - 출력: dX:[M,N], (dgamma:[N]?, dbeta:[N]?)
 * gamma/beta가 null이면 scale/bias 없는 LN으로 처리
 */
Status LayerNormCudaBackwardLaunch(const Tensor& X,
                                   const Tensor* gamma,       // null 가능
                                   const Tensor& dY,
                                   Tensor& dX,
                                   Tensor* dgamma,            // null 가능
                                   Tensor* dbeta,             // null 가능
                                   const LayerNormAttrs& attrs,
                                   StreamHandle stream);

} // namespace ai
