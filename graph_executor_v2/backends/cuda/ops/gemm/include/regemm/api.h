#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>  // for cudaStream_t

namespace regemm {

// -------------------- Enums (기존 유지) --------------------
enum class ActKind : int {
  None      = 0,
  ReLU      = 1,
  LeakyReLU = 2,
  GELU      = 3, // tanh approximation
  Sigmoid   = 4,
  Tanh      = 5,
};

enum class BiasKind : int {
  None   = 0, // bias == nullptr
  Scalar = 1, // single scalar
  PerM   = 2, // one per row (M)
  PerN   = 3  // one per col (N)
};

// -------------------- 기존 파라미터/런처 (호환 유지) --------------------
struct GemmBiasActParams {
  // Dims
  int M = 0, N = 0, K = 0;

  // A: [M x K], row-major, lda >= K
  const void* A = nullptr; int lda = 0;

  // B: [K x N], row-major, ldb >= N
  const void* B = nullptr; int ldb = 0;

  // C: [M x N], optional, row-major, ldc >= N
  const void* C = nullptr; int ldc = 0;

  // D: [M x N], output, row-major, ldd >= N
  void* D = nullptr; int ldd = 0;

  // Scales
  float alpha = 1.0f; // multiplies (A@B)
  float beta  = 0.0f; // multiplies C

  // Bias
  const void* bias = nullptr; // nullptr if none
  BiasKind bias_kind = BiasKind::None;

  // Activation
  ActKind act = ActKind::None;
  float leaky_slope = 0.01f;
};

// Host launchers (implemented in launcher.cu)
void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);
void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s);

// -------------------- 확장 Forward: Z Stash 지원 --------------------
//  - 목적: activation 직전의 pre-activation(Z)을 선택적으로 별도 버퍼에 저장(save_preact)
//  - Z의 leading dim은 D와 동일(ldd)로 통일(실사용 시 layout 혼동 방지)
//  - NOTE: Z는 D와 alias 가능. pre를 Z에 먼저 기록하고, 그 뒤 act(pre)를 D에 기록하므로 안전.
struct GemmBiasActParamsEx {
  // Dims
  int M = 0, N = 0, K = 0;

  // A: [M x K], lda >= K
  const void* A = nullptr; int lda = 0;

  // B: [K x N], ldb >= N
  const void* B = nullptr; int ldb = 0;

  // C: [M x N], optional, ldc >= N (C == nullptr 이면 미사용)
  const void* C = nullptr; int ldc = 0;

  // D: [M x N], output, ldd >= N
  void* D = nullptr; int ldd = 0;

  // Scales
  float alpha = 1.0f; // multiplies (A@B)
  float beta  = 0.0f; // multiplies C (C != nullptr일 때만 의미)

  // Bias
  const void* bias = nullptr; // nullptr이면 bias 없음
  BiasKind bias_kind = BiasKind::None;

  // Activation
  ActKind act = ActKind::None;
  float   leaky_slope = 0.01f; // LeakyReLU용

  // --- NEW: pre-activation(Z) stash ---
  // Z: [M x N], optional, ldZ == ldd 권장(동일 레이아웃)
  void* Z = nullptr;
  int   ldZ = 0;          // 0이면 내부에서 ldd로 간주
  int   save_preact = 0;  // 1이면 activation 직전 값(Z)을 Z 버퍼에 기록
};

// Forward(EX) — bufs 파싱은 런처에서 하되, 구조체로 전달하는 정식 API
void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s);

// -------------------- Backward: 완전 교체용 API --------------------
// 수식:
//   Z = alpha*(A@B) + beta*C + bias,    Y = act(Z)
// 입력: A, B, (C), gY, Z
// 출력: gA, gB, (gC), (gBias)
//  * gC = beta * gZ  (C 사용 시)
//  * gBias:
//      Scalar -> sum(gZ)
//      PerM   -> sum(gZ, axis=1)  (size M)
//      PerN   -> sum(gZ, axis=0)  (size N)
struct GemmBiasActBwdParams {
  // Dims
  int M = 0, N = 0, K = 0;

  // A,B (forward inputs)
  const void* A = nullptr; int lda = 0;  // [M x K], lda >= K
  const void* B = nullptr; int ldb = 0;  // [K x N], ldb >= N

  // C (optional, forward input)
  const void* C = nullptr; int ldc = 0;  // [M x N], C==nullptr이면 미사용

  // Grad at output (gY) and pre-activation stash (Z)
  const void* gY = nullptr; int ldgY = 0; // [M x N], ldgY >= N
  const void* Z  = nullptr; int ldZ  = 0; // [M x N], ldZ  >= N (forward에서 저장됨)

  // Output grads
  void* gA = nullptr; int ldgA = 0;      // [M x K]
  void* gB = nullptr; int ldgB = 0;      // [K x N]
  void* gC = nullptr; int ldgC = 0;      // [M x N], optional (C 사용 시)
  void* gBias = nullptr;                 // scalar / [M] / [N], bias_kind에 따라

  // Hyper-params (forward와 동일해야 일관)
  float alpha = 1.0f;
  float beta  = 0.0f;
  BiasKind bias_kind = BiasKind::None;
  ActKind  act       = ActKind::None;
  float    leaky_slope = 0.01f;

  // --- NEW: optional scratch to avoid per-call cudaMalloc/free ---
  //  외부에서 gZ 임시 버퍼(연속, ld=N)를 제공하면 내부 할당을 건너뜀.
  //  제공하지 않거나 ldgZ!=N이면 내부에서 안전하게 할당.
  float* gZ_scratch = nullptr;
  int    ldgZ       = 0;  // 제공 시 반드시 N이어야 함(가드 목적)
};

// Backward(EX)
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);

} // namespace regemm
