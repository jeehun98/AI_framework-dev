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
  int M = 0, N = 0, K = 0;
  const void* A = nullptr; int lda = 0;
  const void* B = nullptr; int ldb = 0;
  const void* C = nullptr; int ldc = 0;
  void*       D = nullptr; int ldd = 0;
  float alpha = 1.0f;
  float beta  = 0.0f;
  const void* bias = nullptr;
  BiasKind bias_kind = BiasKind::None;
  ActKind act = ActKind::None;
  float   leaky_slope = 0.01f;
};

void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);
void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s);

// -------------------- 확장 Forward: Z Stash 지원 --------------------
struct GemmBiasActParamsEx {
  int M = 0, N = 0, K = 0;

  const void* A = nullptr; int lda = 0;  // [M x K], lda >= K
  const void* B = nullptr; int ldb = 0;  // [K x N], ldb >= N
  const void* C = nullptr; int ldc = 0;  // [M x N], optional

  void* D = nullptr; int ldd = 0;        // [M x N], output

  float alpha = 1.0f;
  float beta  = 0.0f;

  const void* bias = nullptr;
  BiasKind bias_kind = BiasKind::None;

  ActKind act = ActKind::None;
  float   leaky_slope = 0.01f;

  // --- pre-activation stash ---
  void* Z   = nullptr;   // [M x N], optional
  int   ldZ = 0;         // 0 -> treat as ldd
  int   save_preact = 0; // 1: write pre-activation into Z

  // --- NEW: Lt workspace (옵션) ---
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;
};

void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s);

// -------------------- Backward --------------------
struct GemmBiasActBwdParams {
  int M = 0, N = 0, K = 0;

  // Forward inputs
  const void* A = nullptr; int lda = 0;  // [M x K]
  const void* B = nullptr; int ldb = 0;  // [K x N]
  const void* C = nullptr; int ldc = 0;  // [M x N], optional

  // Grad / stash
  const void* gY = nullptr; int ldgY = 0; // [M x N]
  const void* Z  = nullptr; int ldZ  = 0; // [M x N] (pre-activation)

  // Output grads
  void* gA = nullptr; int ldgA = 0;      // [M x K]
  void* gB = nullptr; int ldgB = 0;      // [K x N]
  void* gC = nullptr; int ldgC = 0;      // [M x N], optional
  void* gBias = nullptr;                 // scalar/[M]/[N], bias_kind에 따름

  // Hyper-params
  float    alpha = 1.0f;
  float    beta  = 0.0f;
  BiasKind bias_kind = BiasKind::None;
  ActKind  act       = ActKind::None;
  float    leaky_slope = 0.01f;

  // --- NEW: capture-safe scratch (dZ) ---
  //  외부에서 gZ 임시 버퍼(=dZ)를 주면 내부 malloc/free 없이 사용
  float* gZ_scratch = nullptr; // size >= M*N (row-major, ld == N)
  int    ldgZ       = 0;       // 반드시 N (가드용). 0이면 내부에서 N으로 취급/검증.

  // --- NEW: Lt workspace (옵션) ---
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;
};

// Backward (확장 버전; 이름은 호환 유지)
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);

} // namespace regemm
