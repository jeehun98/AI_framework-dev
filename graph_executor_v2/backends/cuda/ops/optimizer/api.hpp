#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"  // Tensor, Status, StreamHandle 등
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

// -------------------- SGD --------------------
struct SGDAttrs {
  float lr{1e-2f};         // 학습률 (>0)
  float momentum{0.0f};    // [0,1)
  float dampening{0.0f};   // [0,1)
  bool  nesterov{false};   // 네스테로프 사용
  float weight_decay{0.0f}; // L2 decay (>=0)
};

// In-place 업데이트
// - P: 파라미터 [N] (float32, CUDA, 1D)
// - G: 기울기   [N] (float32, CUDA, 1D)
// - V: 모멘텀 버퍼 [N] (momentum>0일 때 필수, 아니면 nullptr 허용)
Status SGDCudaUpdateLaunch(Tensor& P,
                           const Tensor& G,
                           Tensor* V,
                           const SGDAttrs& attrs,
                           StreamHandle stream);

// -------------------- AdamW --------------------
struct AdamWAttrs {
  float lr{1e-3f};          // 학습률 (>0)
  float beta1{0.9f};        // [0,1)
  float beta2{0.999f};      // [0,1)
  float eps{1e-8f};         // >0
  float weight_decay{0.0f}; // decoupled (>=0)
  bool  bias_correction{true}; // m,v에 대한 t 보정 사용 여부
  int   step{1};            // t(스텝) >= 1
};

// In-place 업데이트
// - P: 파라미터 [N]
// - G: 기울기   [N]
// - M: 1차 모멘트 [N]
// - V: 2차 모멘트 [N]
Status AdamWCudaUpdateLaunch(Tensor& P,
                             const Tensor& G,
                             Tensor& M,
                             Tensor& V,
                             const AdamWAttrs& attrs,
                             StreamHandle stream);

} // namespace ai
