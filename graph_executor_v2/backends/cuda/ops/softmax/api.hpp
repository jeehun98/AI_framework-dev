// backends/cuda/ops/softmax/api.hpp
#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"  // Tensor, Status, StreamHandle, DType, Device 등
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

namespace ai {

/**
 * @brief Softmax/LogSoftmax 런처 속성
 *
 * y = softmax(scale * x) 또는 log_softmax(scale * x)
 * - scale = 1/T 역할 (temperature scaling)
 * - log   = true면 log_softmax
 */
struct SoftmaxAttrs {
  float scale{1.0f};   ///< y = softmax(scale * x)
  bool  log{false};    ///< true -> log_softmax
};

/**
 * @brief Row-wise Softmax/LogSoftmax (2D)
 *
 * 입력/마스크/출력의 기대 형태:
 * - X: [M, N] (row-major, 연속 가정 권장)
 * - Mask: null 가능. 있으면 X에 더해짐(additive mask). 크기 선택지:
 *     [M, N] 또는 브로드캐스트 가능한 [1, N] / [M, 1]
 *     (예: -inf 마스킹 또는 0 가중치 덧셈)
 * - Y: [M, N]
 *
 * 안정성:
 * - 내부적으로 row-wise max-shift 사용 (log-sum-exp 안정화)
 *
 * 자료형:
 * - F32 권장. (추후 F16/BF16 지원 시 내부 변환 가능)
 */
Status SoftmaxCudaLaunch(const Tensor& X,
                         const Tensor* Mask,   // null 가능; X에 더해짐(예: -inf 또는 0)
                         Tensor& Y,
                         const SoftmaxAttrs& attrs,
                         StreamHandle stream);

/**
 * @brief Backward: dY -> dX (row-wise)
 *
 * 두 가지 경로를 지원:
 * 1) y_provided = true  : Y_or_X = Y (forward 출력) 를 이용 → 재계산 방지로 빠름/안정
 * 2) y_provided = false : Y_or_X = X (입력) 를 이용 → 내부에서 forward 1회 재계산
 *
 * 입력/출력:
 * - Y_or_X : Y(권장) 또는 X, 크기 [M, N]
 * - Mask   : null 가능. forward 때와 동일하게 적용(브로드캐스트 허용)
 * - dY     : [M, N]
 * - dX     : [M, N]
 *
 * 주의:
 * - attrs.log == true 인 경우에도 동일 API (log_softmax의 미분 규칙 적용)
 */
Status SoftmaxCudaBackwardLaunch(const Tensor& Y_or_X, // Y(권장) 또는 X
                                 const Tensor* Mask,   // null 가능 (forward와 동일 규칙)
                                 const Tensor& dY,
                                 Tensor& dX,
                                 const SoftmaxAttrs& attrs,
                                 bool y_provided,      // true면 첫 인자는 Y
                                 StreamHandle stream);

} // namespace ai
