#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

struct Upsample2DAttrs {
  int   out_h{0}, out_w{0};    // out_h/out_w가 지정되면 우선
  float scale_h{0.f}, scale_w{0.f}; // 없으면 scale 사용
  bool  align_corners{false};
};

// Nearest (기존)
Status Upsample2DNearestCudaLaunch(const Tensor& X, Tensor& Y,
                                   const Upsample2DAttrs& attrs,
                                   StreamHandle stream);
Status Upsample2DNearestBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                           const Upsample2DAttrs& attrs,
                                           StreamHandle stream);

// ✅ Bilinear (신규)
Status Upsample2DBilinearCudaLaunch(const Tensor& X, Tensor& Y,
                                    const Upsample2DAttrs& attrs,
                                    StreamHandle stream);
Status Upsample2DBilinearBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                            const Upsample2DAttrs& attrs,
                                            StreamHandle stream);

} // namespace ai
