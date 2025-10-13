// backends/cuda/ops/layernorm/api.hpp
#pragma once
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct LayerNormAttrs { float eps{1e-5f}; };

// 캡처-세이프 선택 워크스페이스 (없어도 동작)
struct LayerNormWorkspaceFwd {
  float* mean    {nullptr};  // [M]
  float* inv_std {nullptr};  // [M]
};
struct LayerNormWorkspaceBwd {
  float* sum_dy      {nullptr}; // [M]
  float* sum_dy_xhat {nullptr}; // [M]
};

/**
 * Forward: Y = LN(X; gamma, beta)
 * - gamma/beta는 null 가능
 * - ws_fwd는 null 가능 (제공 시 row-wise 통계 저장/재사용)
 */
Status LayerNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,   // null 가능
                           const Tensor* beta,    // null 가능
                           Tensor& Y,
                           const LayerNormAttrs& attrs,
                           StreamHandle stream,
                           const LayerNormWorkspaceFwd* ws_fwd /*=nullptr*/);

/**
 * Backward:
 *  입력 : X:[M,N], dY:[M,N], (gamma:[N]?)
 *  출력 : dX:[M,N], (dgamma:[N]?, dbeta:[N]?)
 *  - ws_bwd는 null 가능 (제공 시 row-wise 보조 리덕션 저장)
 */
Status LayerNormCudaBackwardLaunch(const Tensor& X,
                                   const Tensor* gamma,       // null 가능
                                   const Tensor& dY,
                                   Tensor& dX,
                                   Tensor* dgamma,            // null 가능
                                   Tensor* dbeta,             // null 가능
                                   const LayerNormAttrs& attrs,
                                   StreamHandle stream,
                                   const LayerNormWorkspaceBwd* ws_bwd /*=nullptr*/);

} // namespace ai
