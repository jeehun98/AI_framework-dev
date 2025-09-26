#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

// Scaled Dot-Product Attention attributes
struct SDPAAttrs {
  float     scale{0.f};       // 0이면 1/sqrt(D)로 자동
  bool      causal{false};    // true면 i<j 마스크(후속 PR에서 커널화)
  float     dropout_p{0.f};   // 0~1 미만
  bool      scale_in_train{true};
  uint64_t  seed{0};          // dropout seed
};

// FWD: Q:[B,H,M,D], K:[B,H,N,D], V:[B,H,N,D] -> Y:[B,H,M,D]
// mask(opt): [B,1,M,N] (I8/I32: 0=keep,1=mask) 또는 F32(−inf/0)
Status SDPACudaLaunch(const Tensor& Q,
                      const Tensor& K,
                      const Tensor& V,
                      const Tensor* mask,   // nullable
                      Tensor& Y,
                      const SDPAAttrs& attrs,
                      StreamHandle stream);

// 변경: mask 추가
Status SDPACudaBackwardLaunch(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& dY,
                              const Tensor* mask,              // <-- 추가
                              Tensor* dQ, Tensor* dK, Tensor* dV,
                              const SDPAAttrs& attrs,
                              StreamHandle stream);

} // namespace ai
