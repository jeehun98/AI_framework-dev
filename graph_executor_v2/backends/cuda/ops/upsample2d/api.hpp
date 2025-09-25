#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

struct Upsample2DAttrs {
  // 출력 크기(out_h/out_w) 또는 scale_h/scale_w 중 하나만 지정
  int   out_h{0}, out_w{0};  // 우선권 높음 (설정되면 이것 사용)
  float scale_h{0.f}, scale_w{0.f}; // out_h/out_w가 0이면 scale 사용
  bool  align_corners{false};       // PyTorch 호환 규칙
};

// Nearest Neighbor
Status Upsample2DNearestCudaLaunch(const Tensor& X, Tensor& Y,
                                   const Upsample2DAttrs& attrs,
                                   StreamHandle stream);

Status Upsample2DNearestBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                           const Upsample2DAttrs& attrs,
                                           StreamHandle stream);

} // namespace ai
