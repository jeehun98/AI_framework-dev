// backends/cuda/ops/pool2d/api.hpp
#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct Pool2DAttrs {
  int kH{2}, kW{2};
  int sH{2}, sW{2};
  int pH{0}, pW{0};
  int dH{1}, dW{1};
  bool ceil_mode{false};      // 출력 크기 계산 시 ceil 적용
  bool count_include_pad{false}; // AvgPool 전용 (여기선 false 권장)
};

// MaxPool: FWD는 선택적으로 Indices(int32) 출력
Status MaxPool2DCudaLaunch(const Tensor& X, Tensor& Y, Tensor* Indices,
                           const Pool2DAttrs& attrs, StreamHandle stream);
Status MaxPool2DBackwardCudaLaunch(const Tensor& dY, const Tensor& Indices, Tensor& dX,
                                   const Pool2DAttrs& attrs, StreamHandle stream);

// AvgPool: 인덱스 필요 없음
Status AvgPool2DCudaLaunch(const Tensor& X, Tensor& Y,
                           const Pool2DAttrs& attrs, StreamHandle stream);
Status AvgPool2DBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                   const Pool2DAttrs& attrs, StreamHandle stream);

} // namespace ai
