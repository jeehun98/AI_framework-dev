#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

// -------------------------
// Legacy/regular CE enums & attrs (분리형 커널)
// -------------------------
enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };

struct CrossEntropyAttrs {
  bool from_logits{true};
  Reduction reduction{Reduction::Mean};
  int   ignore_index{-1};   // -1이면 비활성
  float eps{1e-9f};         // from_logits=false일 때 log(p+eps) 안정화
  float ls_eps{0.f};        // label smoothing (0=off), 권장 0.1~0.2
};

// X:[M,N], target:[M] (int32) -> loss
//   reduction=None    => loss:[M]
//   reduction=Mean/Sum=> loss:[1]
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

// -------------------------
// Fused Softmax + CE (from logits) attrs & API
//   - 이번 커널 버전은 ignore_index / label smoothing 미지원
//   - 지원: stable LSE softmax, reduction(None/Mean/Sum)
// -------------------------
struct SCEFuseAttrs {
  enum class Reduction : int { None = 0, Mean = 1, Sum = 2 };
  Reduction reduction{Reduction::Mean};
  bool stable{true};  // log-sum-exp 안정화 사용 여부
};

// logits:[M,C] F32, labels:[M] I32 → (dlogits, optional loss)
// - dlogits:[M,C] F32 (필수, in-place 아님)
// - loss: reduction==None → [M], Mean/Sum → [1]; nullptr 허용
Status SoftmaxCEFusedForwardBackwardCudaLaunch(const Tensor& logits,
                                               const Tensor& labels,
                                               Tensor& dlogits,
                                               Tensor* loss,                 // optional
                                               const SCEFuseAttrs& attrs,
                                               StreamHandle stream);

} // namespace ai
