#pragma once
/**
 * @file api.hpp
 * @brief GEMM (CUDA) forward/backward API — Z(pre-activation) 저장 옵션 지원
 */

#include "backends/cuda/ops/_common/shim/ai_shim.hpp" // Tensor, GemmAttrs, StreamHandle, Status

namespace ai {

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
 */
ai::Status GemmCudaLaunch(
    const Tensor& A,
    const Tensor& B,
    const Tensor* Bias,
    Tensor&       Y,
    const GemmAttrs& attrs,
    StreamHandle  stream,
    Tensor*       Z_saved /* = nullptr */
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
 */
ai::Status GemmCudaBackward(
    const Tensor& A,
    const Tensor& B,
    const Tensor* C,
    const Tensor& gY,
    const Tensor& Z,
    Tensor* gA,
    Tensor* gB,
    Tensor* gC,
    Tensor* gBias,
    const GemmAttrs& attrs,
    StreamHandle stream
);

} // namespace ai
