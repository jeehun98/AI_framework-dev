// backends/cuda/ops/gemm/api.hpp

#pragma once
/**
 * @file api.hpp
 * @brief GEMM (CUDA) forward/backward API — Z(pre-activation) 저장 + capture-safe workspace 지원
 */
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include <cstddef>

namespace ai {

struct GemmWorkspace {
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;
  void*  scratch            = nullptr;   // e.g., dZ buffer for backward
  size_t scratch_bytes      = 0;
};

// ---- 정식 선언: 기본인자는 맨 끝 2개만 ----
ai::Status GemmCudaLaunch(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    Bias,          // ← 기본값 주지 마세요!
    Tensor&          Y,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    Tensor*          Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
);

// ---- 편의 오버로드: Bias 생략 시 nullptr 사용 ----
inline ai::Status GemmCudaLaunch(
    const Tensor&    A,
    const Tensor&    B,
    Tensor&          Y,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    Tensor*          Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
) {
  return GemmCudaLaunch(A, B, /*Bias=*/nullptr, Y, attrs, stream, Z_saved, ws);
}

// ---- Backward도 동일 원칙 ----
ai::Status GemmCudaBackward(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    C,              // 기본값 X
    const Tensor&    gY,
    const Tensor&    Z,
    Tensor*          gA,             // 기본값 X (포인터는 호출부에서 nullptr 전달)
    Tensor*          gB,
    Tensor*          gC,
    Tensor*          gBias,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    const GemmWorkspace* ws = nullptr
);

// 하위호환 편의 래퍼(원하시면 유지)
inline ai::Status GemmCudaBackwardInto(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    C,
    const Tensor&    gY,
    const Tensor&    Z,
    Tensor*          gA,
    Tensor*          gB,
    Tensor*          gC,
    Tensor*          gBias,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    float*           dZ,
    void*            lt_ws      = nullptr,
    size_t           lt_ws_bytes = 0)
{
    GemmWorkspace ws{};
    ws.scratch            = static_cast<void*>(dZ);
    ws.scratch_bytes      = 0;
    ws.lt_workspace       = lt_ws;
    ws.lt_workspace_bytes = lt_ws_bytes;

    return GemmCudaBackward(
        A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream, &ws
    );
}

} // namespace ai
