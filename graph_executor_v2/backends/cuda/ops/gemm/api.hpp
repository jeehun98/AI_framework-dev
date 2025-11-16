#pragma once
/**
 * @file api.hpp
 * @brief GEMM (CUDA) forward/backward API — Z(pre-activation) 저장 + capture-safe workspace 지원
 */

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"  // Tensor/GemmAttrs/Status/Stream 등
#include "backends/cuda/ops/_common/shim/enums.hpp"     // (안전) ActKind 등
#include "backends/cuda/ops/_common/shim/traits.hpp"    // ★ BiasMode, to_bias_mode 등

#include <cstddef>
#include <cstdint>

namespace ai::cuda::shim {

// =====================================================================
//  ▷ 커널 파라미터 (FWD)
// =====================================================================
struct GemmBiasActParams {
  int M, N, K;
  const void* A; int lda;
  const void* B; int ldb;
  const void* C; int ldc;   // optional (HasC 경로)
  void*       D; int ldd;

  float alpha;              // D = act( alpha * (A@B) + beta * C + bias )
  float beta;

  const void* bias;         // nullptr이면 BiasKind::None
  BiasKind    bias_kind;

  ActKind act;
  float   leaky_slope;
};

struct GemmBiasActParamsEx {
  int M, N, K;
  const void* A; int lda;
  const void* B; int ldb;
  const void* C; int ldc;   // optional (HasC 경로)
  void*       D; int ldd;

  float alpha, beta;

  const void* bias;
  BiasKind    bias_kind;

  ActKind act;
  float   leaky_slope;

  // Z(pre-activation) 저장 옵션
  void* Z; int ldZ;         // ldZ==0이면 ldd 사용
  int   save_preact;        // 0/1

  // (선택) cuBLASLt 등 외부 워크스페이스 전달
  void*       lt_workspace{nullptr};
  std::size_t lt_workspace_bytes{0};
};

// =====================================================================
//  ▷ 커널 파라미터 (BWD)
// =====================================================================
struct GemmBiasActBwdParams {
  int M, N, K;

  // --- FWD 입력 ---
  const void* A;  int lda;
  const void* B;  int ldb;
  const void* C;  int ldc;     // optional
  const void* Z;  int ldZ;     // pre-activation
  const void* gY; int ldgY;    // dLoss/dY (outgrad)

  // --- 출력 ---
  void* gA; int ldgA;          // dLoss/dA (nullable)
  void* gB; int ldgB;          // dLoss/dB (nullable)
  void* gC; int ldgC;          // dLoss/dC (nullable)
  void* gBias;                 // dLoss/dbias (nullable)

  // --- 임시 버퍼 ---
  float* gZ_scratch; int ldgZ; // gZ 저장(dLoss/dZ), ld = N 가정 (nullable)

  // --- 스칼라 파라미터 ---
  float    alpha{1.f};
  float    beta{0.f};          // gC 계산 시 사용
  BiasKind bias_kind{BiasKind::None};
  ActKind  act{ActKind::None};
  float    leaky_slope{0.01f};

  // --- (선택) 외부 워크스페이스 전달 ---
  void*       lt_workspace{nullptr};
  std::size_t lt_workspace_bytes{0};
};

// =====================================================================
//  ▷ 공용 워크스페이스/헬퍼
// =====================================================================
struct alignas(256) GemmWorkspace {
  void*       lt_workspace       = nullptr;
  std::size_t lt_workspace_bytes = 0;

  // Backward용 임시 gZ 버퍼 (row-major, ld = N 가정)
  void*       scratch            = nullptr;
  std::size_t scratch_bytes      = 0;
};

[[nodiscard]] constexpr std::size_t
GemmRequiredBackwardScratchBytes(std::int64_t M, std::int64_t N) noexcept {
  return (M <= 0 || N <= 0)
           ? 0ull
           : static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * sizeof(float);
}

[[nodiscard]] inline bool
GemmIsLtWorkspaceAligned(const void* p, std::size_t alignment = 256) noexcept {
  return is_workspace_aligned(p, alignment);
}

// =====================================================================
//  ▷ API (FWD/BWD)
// =====================================================================

// Forward: A[M,K] @ B[K,N] + bias -> Y[M,N]
//  - attrs.save_z==true 이면 Z_saved에 pre-activation 저장
Status GemmCudaLaunch(
    const Tensor&     A,
    const Tensor&     B,
    const Tensor*     Bias,
    Tensor&           Y,
    const GemmAttrs&  attrs,
    StreamHandle      stream,
    Tensor*           Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
);

// Bias 인자가 없는 편의 오버로드
inline Status GemmCudaLaunch(
    const Tensor&     A,
    const Tensor&     B,
    Tensor&           Y,
    const GemmAttrs&  attrs,
    StreamHandle      stream,
    Tensor*           Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
) {
  return GemmCudaLaunch(A, B, /*Bias=*/nullptr, Y, attrs, stream, Z_saved, ws);
}

// Backward: (A,B,C,gY,Z) -> (gA,gB,gC,gBias)
//  - ws가 주어지면 capture-safe 경로 (내부 malloc 없음)
Status GemmCudaBackward(
    const Tensor&     A,
    const Tensor&     B,
    const Tensor*     C,
    const Tensor&     gY,
    const Tensor&     Z,
    Tensor*           gA,
    Tensor*           gB,
    Tensor*           gC,
    Tensor*           gBias,
    const GemmAttrs&  attrs,
    StreamHandle      stream,
    const GemmWorkspace* ws = nullptr
);

// Capture-safe 편의 래퍼(dZ/gZ_scratch 직접 전달)
inline Status GemmCudaBackwardInto(
    const Tensor&     A,
    const Tensor&     B,
    const Tensor*     C,
    const Tensor&     gY,
    const Tensor&     Z,
    Tensor*           gA,
    Tensor*           gB,
    Tensor*           gC,
    Tensor*           gBias,
    const GemmAttrs&  attrs,
    StreamHandle      stream,
    float*            dZ,
    void*             lt_ws       = nullptr,
    std::size_t       lt_ws_bytes = 0)
{
  GemmWorkspace ws{};
  ws.scratch            = static_cast<void*>(dZ);
  ws.scratch_bytes      = 0; // 내부에서 필요 크기 검증
  ws.lt_workspace       = lt_ws;
  ws.lt_workspace_bytes = lt_ws_bytes;
  return GemmCudaBackward(A,B,C,gY,Z,gA,gB,gC,gBias,attrs,stream,&ws);
}

} // namespace ai::cuda::shim


// =====================================================================
//  ▷ 커널/런처 선언 (config.h는 포함하지 않음; 구현부에서 include)
//     시그니처는 launcher.cu 호출부와 반드시 동일해야 함
// =====================================================================
namespace ai::cuda::shim {

// (FWD) 타일드 EX 커널 — 단일 packed params 인자
template<int BM_, int BN_, int BK_,
         ActKind AK, BiasMode BM, bool HasC, bool SaveZ>
__global__ void gemm_bias_act_f32_tiled_kernel_ex(GemmBiasActParamsEx p);

// (FWD) 작은 문제용 스모크 런처
void launch_gemm_bias_act_f32_smoke_ex(const GemmBiasActParamsEx& p, cudaStream_t s);

// (BWD) 메인 백워드 런처
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);

} // namespace ai::cuda::shim
