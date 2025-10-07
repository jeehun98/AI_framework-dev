#pragma once
/**
 * @file api.hpp
 * @brief GEMM (CUDA) forward/backward API — Z(pre-activation) 저장 + capture-safe workspace 지원
 */

#include "backends/cuda/ops/_common/shim/ai_shim.hpp" // Tensor, GemmAttrs, StreamHandle, Status
#include <cstddef>

namespace ai {

/**
 * @brief 공용 워크스페이스 (fwd/bwd 공통)
 *
 * - lt_workspace       : (선택) cublasLt workspace 버퍼
 * - lt_workspace_bytes : lt_workspace 바이트 크기
 * - scratch            : (선택) 추가 스크래치 버퍼
 *   * bwd에서 dZ(M*N, float) 용도로 사용 권장 (그래프 캡처 시 필수)
 * - scratch_bytes      : 스크래치 크기 (일부는 내부에서 shape 기반으로 검증)
 */
struct GemmWorkspace {
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;

  void*  scratch            = nullptr;   // e.g., dZ buffer for backward
  size_t scratch_bytes      = 0;
};

/**
 * @brief GEMM + (Bias) + Act Forward
 *
 * 전제:
 *  - dtype: f32, layout: RowMajor
 *  - alpha=1, beta=0 (현재 스펙)
 *  - (trans_a|trans_b)==false
 *
 * 동작:
 *  - attrs.save_z == true && Z_saved != nullptr:
 *      Z_saved ← (A @ B [+ Bias])  // pre-activation 저장
 *      Y       ← act(Z_saved)       // act=None이면 Y ← Z_saved (복사/alias 가능)
 *  - 그 외: 기존 fused 경로 사용 (Z 미저장)
 *
 * @param ws (선택): capture-safe / 튜닝용 워크스페이스
 */
ai::Status GemmCudaLaunch(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    Bias,
    Tensor&          Y,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    Tensor*          Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
);

/**
 * @brief GEMM + (Bias) + Act Backward
 *
 * 입력:
 *  - A:(M,K), B:(K,N), (선택)C:(M,N), gY:(M,N), Z:(M,N; pre-activation)
 * 출력(옵션):
 *  - gA:(M,K), gB:(K,N), gC:(M,N), gBias:(Scalar/PerM/PerN)
 * 전제:
 *  - dtype: f32, layout: RowMajor
 *  - (trans_a|trans_b)==false
 *
 * @param ws (선택): 워크스페이스
 *   * 그래프 캡처 안전 경로에서는 ws->scratch를 dZ(float[M*N]) 용도로 제공해야 함
 *   * 크기 검증(M*N*sizeof(float))은 내부에서 shape 기반으로 수행
 */
ai::Status GemmCudaBackward(
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
    const GemmWorkspace* ws = nullptr
);

/* (선택) 하위호환: 과거 *_Into 호출부를 위해 얇은 인라인 래퍼 제공.
   만약 이전에 .cu에 별도 정의가 있었다면 제거하고, 아래 인라인만 남겨 ODR 충돌 방지. */
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
    float*           dZ,                // required: float[M*N]
    void*            lt_ws      = nullptr,
    size_t           lt_ws_bytes = 0)
{
    GemmWorkspace ws{};
    ws.scratch            = static_cast<void*>(dZ);
    ws.scratch_bytes      = 0;          // 내부에서 M*N*sizeof(float)로 검증
    ws.lt_workspace       = lt_ws;
    ws.lt_workspace_bytes = lt_ws_bytes;

    return GemmCudaBackward(
        A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream, &ws
    );
}

} // namespace ai
