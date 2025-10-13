#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"  // Tensor, Status, StreamHandle, DType, Device 등
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp" // Status, StreamHandle
#endif

#include <cstdint>

namespace ai {

// ============================ Softmax / LogSoftmax ============================

/**
 * @brief Softmax/LogSoftmax 런처 속성
 * y = softmax(scale * x) 또는 log_softmax(scale * x)
 * - scale = 1/T (temperature scaling)
 * - log   = true면 log_softmax
 */
struct SoftmaxAttrs {
  float scale{1.0f};
  bool  log{false};
};

/**
 * @brief (선택) 캡처-세이프 워크스페이스
 * 큰 M(행 개수)에서 성능 최적화를 위해 행별 통계를 미리 저장할 버퍼.
 * 제공하지 않으면 내부 공유메모리/두-패스 등으로 동작.
 */
struct SoftmaxWorkspaceFwd {
  // 크기: [M]
  float* row_max{nullptr};   // 선택: 각 행의 max(x)
  float* row_sum{nullptr};   // 선택: 각 행의 sum(exp(shifted))
};

struct SoftmaxWorkspaceBwd {
  // 크기: [M]
  float* row_dot{nullptr};   // 선택: 각 행의 dot(dY, Y) (softmax bwd에서 사용)
};

/**
 * @brief Row-wise Softmax/LogSoftmax (2D)
 *
 * 입력/출력:
 * - X:    [M, N] (F32)
 * - Mask: null 가능. additive mask (브로드캐스트 허용: [M,N], [1,N], [M,1])
 * - Y:    [M, N]
 *
 * 주의: 내부 동적할당/Host↔Device memcpy 없음(캡처-세이프).
 */
Status SoftmaxCudaLaunch(const Tensor& X,
                         const Tensor* Mask,   // null 허용
                         Tensor& Y,
                         const SoftmaxAttrs& attrs,
                         StreamHandle stream,
                         const SoftmaxWorkspaceFwd* ws_fwd /*=nullptr*/ = nullptr);

/**
 * @brief Backward: dY -> dX (row-wise)
 *
 * y_provided:
 * - true  : 첫 인자 Y_or_X는 Y (forward 출력), 재계산 없음
 * - false : 첫 인자 Y_or_X는 X, 내부에서 1회 fwd 재계산
 *
 * 입출력:
 * - Y_or_X : [M, N]
 * - Mask   : null 가능 (forward와 동일 규칙)
 * - dY     : [M, N]
 * - dX     : [M, N]
 */
Status SoftmaxCudaBackwardLaunch(const Tensor& Y_or_X,  // Y(권장) 또는 X
                                 const Tensor* Mask,    // null 가능
                                 const Tensor& dY,
                                 Tensor& dX,
                                 const SoftmaxAttrs& attrs,
                                 bool y_provided,
                                 StreamHandle stream,
                                 const SoftmaxWorkspaceBwd* ws_bwd /*=nullptr*/ = nullptr);

} // namespace ai
