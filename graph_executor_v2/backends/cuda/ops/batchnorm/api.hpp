#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

// BatchNorm CUDA API (NCHW/NHWC 지원)
// - training: 채널별 평균/분산 감소 + running_* 업데이트 + save_* 출력
// - inference: running_* 고정 사용, 감소 없음
// - 혼합정밀: 입력/출력은 FP16/FP32 허용, 내부 축적은 FP32 가정
// - CUDA Graph 캡처 세이프: attrs와 텐서 shape 고정, ws 크기 고정 시 capture-safe

namespace ai {

// ========== 속성 ==========
// Conv2DAttrs 스타일을 참고한 BatchNorm 속성 세트
struct BatchNormAttrs {
  // 데이터 레이아웃
  bool channels_last{false};   // false: NCHW, true: NHWC

  // 알고리즘/수치
  float eps{1e-5f};            // 분산 안정화 epsilon
  float momentum{0.1f};        // running_mean/var 업데이트 계수 (PyTorch 호환)
  bool  training{true};        // true: 학습 경로, false: 추론 경로

  // 선택 사항
  bool  with_affine{true};     // gamma/beta 사용 여부 (false면 정규화만)
  bool  use_welford{true};     // 감소 시 Welford 사용 (수치 안정)
  int   num_groups{1};         // (확장 포인트) GroupNorm 호환용, BN은 1 고정 사용
};

// ========== 캡처-세이프 워크스페이스 ==========
// BN은 큰 워크스페이스가 필요하진 않지만, 안정적인 capture와
// 대규모 텐서에서의 분할 감소를 위해 선택적 버퍼를 둡니다.
struct BatchNormWorkspaceFwd {
  // 채널 축 감소의 부분합/카운트 보관 버퍼 (선택)
  //   sums:   Σ x,  Σ x^2  (layout: [2, C]) 혹은 분리 버퍼 2개 중 하나 사용
  float* partial_sums{nullptr};   // [2 * C]  (0..C-1: sum(x), C..2C-1: sum(x^2))
  int    partial_sums_stride{0};  // 0이면 [2*C] 일렬, 아니면 row-stride

  // (옵션) 타일 분할 시 블록별 중간 결과 보관
  float* blockbuf{nullptr};       // 구현체 정의용 (크기 고정 시 capture-safe)
  size_t blockbuf_elems{0};

  // NOTE: save_mean/invstd는 API 출력 텐서로 외부에서 제공
};

struct BatchNormWorkspaceBwd {
  // 채널 축 감소의 부분합 버퍼 (dgamma/dbeta용)
  float* partial_sums{nullptr};    // [2 * C] (0..C-1: dbeta(=Σ dY), C..2C-1: dgamma_part=Σ (X-μ)*invstd*dY)
  int    partial_sums_stride{0};

  // (옵션) dX 계산용 중간 버퍼 (예: Σ dY, Σ dY*X_hat)
  float* tempbuf{nullptr};
  size_t tempbuf_elems{0};
};

// ========== Forward ==========
// X: [N,C,H,W] or [N,H,W,C]
// gamma/beta: [C] (with_affine=false면 nullptr 허용)
// running_mean/var: [C] (학습 시 갱신, 추론 시 read-only)
// save_mean/save_invstd: [C] (학습 시 bwd용으로 출력, 추론 시 선택적/무시)
// Y: X와 동일 shape
Status BatchNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,         // [C] or nullptr if !with_affine
                           const Tensor* beta,          // [C] or nullptr if !with_affine
                           Tensor* running_mean,        // [C] (in/out when training, else in)
                           Tensor* running_var,         // [C] (in/out when training, else in)
                           Tensor& Y,                   // out: same shape as X
                           const BatchNormAttrs& attrs,
                           StreamHandle stream,
                           Tensor* save_mean /*=nullptr*/,     // [C] (training일 때 out)
                           Tensor* save_invstd /*=nullptr*/,   // [C] (training일 때 out)
                           const BatchNormWorkspaceFwd* ws_fwd /*=nullptr*/);

// ========== Backward ==========
// dY: [N,C,H,W] or [N,H,W,C]
// X:  forward 입력(학습 시 저장해둔 텐서 또는 동일 값 필요)
// gamma: [C] (with_affine=false면 nullptr 허용 → dgamma/dbeta도 nullptr 권장)
// save_mean/save_invstd: [C] (fwd(training)에서 저장된 값 필요)
// dX:  [N,C,H,W] or [N,H,W,C] (선택적, nullptr이면 생략)
// dgamma, dbeta: [C] (선택적, nullptr이면 생략)
Status BatchNormCudaBackwardLaunch(const Tensor& dY,
                                   const Tensor& X,
                                   const Tensor* gamma,           // [C] or nullptr if !with_affine
                                   const Tensor& save_mean,       // [C]
                                   const Tensor& save_invstd,     // [C]
                                   Tensor* dX,                    // out or nullptr
                                   Tensor* dgamma,                // out or nullptr
                                   Tensor* dbeta,                 // out or nullptr
                                   const BatchNormAttrs& attrs,
                                   StreamHandle stream,
                                   const BatchNormWorkspaceBwd* ws_bwd /*=nullptr*/);

} // namespace ai
