#pragma once

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

// ============================================================================
// Batch Normalization CUDA API (NCHW / NHWC)
// - Training: 채널별 mean/var 계산 → running_* EMA 갱신 → save_*(mean, invstd) 기록
// - Inference: running_* 고정 통계로 정규화 (감소/갱신 없음)
// - 혼합정밀: 입력/출력 FP16/FP32 허용, 내부 누산은 FP32 고정
// - CUDA Graph: 동적 할당 금지, shape/attrs/workspace 크기 불변
// ============================================================================

// ------------------------------ 속성(Attrs) ---------------------------------
/** PyTorch와 동등 의미의 파라미터들 */
struct BatchNormAttrs {
  bool  channels_last{false};  // false:NCHW, true:NHWC
  float eps{1e-5f};            // invstd = rsqrt(var + eps)
  float momentum{0.1f};        // running = (1-m)*running + m*batch
  bool  training{true};        // true: 학습 경로, false: 추론 경로
  bool  with_affine{true};     // γ/β 사용 여부
  bool  use_welford{true};     // Welford 감소 사용
  int   num_groups{1};         // BN은 1 (확장용)
};

// ------------------------- Forward용 워크스페이스 ---------------------------
/** 캡처-세이프/대용량 감소용 선택 버퍼 (필요 시만 사용) */
struct BatchNormWorkspaceFwd {
  // 채널별 부분합 (옵션)
  // [0..C-1]: sum(x), [C..2C-1]: sum(x^2)
  float* partial_sums{nullptr};  // 크기: 2*C (연속 배치)
  int    partial_sums_stride{0}; // 2xC 뷰 사용 시 row stride (0=연속)

  // 블록 레벨 임시 버퍼(옵션)
  float* blockbuf{nullptr};
  size_t blockbuf_elems{0};      // 요소 개수 (float 기준)
};

// ------------------------- Backward용 워크스페이스 --------------------------
/** dγ/dβ 감소 및 dX 보조용 임시 버퍼 (옵션) */
struct BatchNormWorkspaceBwd {
  // [0..C-1]: dbeta = Σ dY
  // [C..2C-1]: dgamma_partial = Σ (X-μ)*invstd*dY
  float* partial_sums{nullptr};  // 크기: 2*C
  int    partial_sums_stride{0};

  float* tempbuf{nullptr};
  size_t tempbuf_elems{0};
};

// --------------------------------- 규약 -------------------------------------
// Shapes:
//   X, Y: [N,C,H,W] (channels_last=false) 또는 [N,H,W,C] (true)
//   gamma, beta, running_mean, running_var, save_mean, save_invstd: [C]
// DTypes:
//   X/Y: F16 또는 F32, 파라미터/통계: F32, 내부 누산: F32
// Aliasing:
//   Y는 X와 동일 버퍼 금지. running_*은 학습 시 in/out 가능.
// Affine:
//   with_affine=false → gamma/beta는 nullptr, BWD dγ/dβ도 보통 nullptr 권장
// Save 텐서:
//   training=true → save_mean/save_invstd 필수, backward에서 사용
// CUDA Graph:
//   동적 할당 없음, workspace 크기 고정, shape/attrs 불변

// -------------------------------- Forward -----------------------------------
/**
 * BatchNorm Forward
 * - 학습: 배치 평균/분산 계산 → running_* EMA 업데이트 → save_* 기록
 * - 추론: running_* 사용해 정규화만 수행
 */
Status BatchNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,        // [C] (with_affine=false면 nullptr)
                           const Tensor* beta,         // [C] (with_affine=false면 nullptr)
                           Tensor* running_mean,       // [C] (학습: in/out, 추론: in)
                           Tensor* running_var,        // [C] (학습: in/out, 추론: in)
                           Tensor& Y,                  // 출력: X와 동일 shape/layout
                           const BatchNormAttrs& attrs,
                           StreamHandle stream,
                           Tensor* save_mean /*=nullptr*/,   // [C] (학습 시 필수)
                           Tensor* save_invstd /*=nullptr*/, // [C] (학습 시 필수)
                           const BatchNormWorkspaceFwd* ws_fwd /*=nullptr*/);

// -------------------------------- Backward ----------------------------------
/**
 * BatchNorm Backward
 * - dY, X, (γ), save_mean, save_invstd로부터 dX, dγ, dβ 계산
 */
Status BatchNormCudaBackwardLaunch(const Tensor& dY,
                                   const Tensor& X,
                                   const Tensor* gamma,        // [C] (with_affine=false면 nullptr)
                                   const Tensor& save_mean,    // [C] (Forward 저장본)
                                   const Tensor& save_invstd,  // [C] (Forward 저장본)
                                   Tensor* dX,                 // (옵션) X 미분
                                   Tensor* dgamma,             // (옵션) γ 미분
                                   Tensor* dbeta,              // (옵션) β 미분
                                   const BatchNormAttrs& attrs,
                                   StreamHandle stream,
                                   const BatchNormWorkspaceBwd* ws_bwd /*=nullptr*/);

// ------------------------------ Workspace 질의 ------------------------------
/** Forward에 필요한 추가 workspace 바이트 수 (없으면 0) */
size_t GetFwdWorkspaceBytes(const Tensor& X, const BatchNormAttrs& attrs);

/** Backward에 필요한 추가 workspace 바이트 수 (없으면 0) */
size_t GetBwdWorkspaceBytes(const Tensor& dY, const Tensor& X, const BatchNormAttrs& attrs);

} // namespace ai
