#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
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
};

// FWD: X[N,Cin,H,W], W[Cout,Cin,Kh,Kw], B[Cout]? → Y[N,Cout,Hout,Wout]
Status Conv2DCudaLaunch(const Tensor& X,
                        const Tensor& W,
                        const Tensor* B,
                        Tensor& Y,
                        const Conv2DAttrs& attrs,
                        StreamHandle stream);

// BWD: dY → (선택) dW[Cout,Cin,Kh,Kw], dB[Cout], dX[N,Cin,H,W]
Status Conv2DCudaBackwardLaunch(const Tensor& X,
                                const Tensor& W,
                                const Tensor& dY,
                                Tensor* dW,        // nullable
                                Tensor* dB,        // nullable
                                Tensor* dX,        // nullable
                                const Conv2DAttrs& attrs,
                                StreamHandle stream);

} // namespace ai
