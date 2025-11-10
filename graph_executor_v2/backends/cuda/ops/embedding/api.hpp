// backends/cuda/ops/embedding/api.hpp
#pragma once
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

struct EmbeddingAttrs {
  int   padding_idx{-1};       // <0 이면 미사용
  bool  scale_grad_by_freq{false};
  float out_scale{1.0f};       // 출력에 곱할 스케일 (예: sqrt(d_model))
};

Status EmbeddingCudaLaunch(       // Forward
  const Tensor& Weight,           // [V, D], float32
  const Tensor& Indices,          // [N, L] or [L], int32/int64
  Tensor& Output,                 // [N, L, D] or [L, D], float32
  const EmbeddingAttrs& attrs,
  StreamHandle stream);

Status EmbeddingCudaBackwardLaunch( // Backward
  const Tensor& Indices,            // [N, L] or [L]
  const Tensor& dY,                 // [N, L, D] or [L, D]
  Tensor* dWeight,                  // [V, D], nullable(없으면 생략)
  const EmbeddingAttrs& attrs,
  StreamHandle stream);

} // namespace ai
