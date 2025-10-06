#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct Conv2DAttrs {
  int stride_h{1}, stride_w{1};
  int pad_h{0}, pad_w{0};
  int dil_h{1}, dil_w{1};
  int groups{1}; // MVP: 1만 지원

  // NEW: epilogue
  ai::ActKind act{ai::ActKind::None};
  float       leaky_slope{0.01f};
  bool        with_bias{false};

  // NEW: pre-activation Z 저장
  bool        save_z{false};
};

enum class ConvBiasKind : int {
  None   = 0,
  Scalar = 1,
  PerC   = 2
};

Status Conv2DCudaLaunch(const Tensor& X,
                        const Tensor& W,
                        const Tensor* B,
                        Tensor& Y,
                        const Conv2DAttrs& attrs,
                        StreamHandle stream,
                        Tensor* Z_saved /*=nullptr*/);

Status Conv2DCudaBackwardLaunch(const Tensor& X,
                                const Tensor& W,
                                const Tensor& dY,
                                const Tensor& Z,
                                Tensor* dW,
                                Tensor* dB,
                                Tensor* dX,
                                const Conv2DAttrs& attrs,
                                StreamHandle stream);

} // namespace ai
