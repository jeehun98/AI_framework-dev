// backends/cuda/ops/gemm/detail/api.h
#pragma once
#include <cstdint>           // int32_t, etc.
#include <cstddef>           // size_t
#include <cuda_runtime_api.h> // cudaStream_t

namespace regemm {

// ---------------------------------------------------------------------
// Layout/LD 규칙 (모든 파라미터 공통)
// - Row-major 가정.
// - A: [M x K],  lda >= K (ld==0이면 내부에서 K로 취급 가능)
// - B: [K x N],  ldb >= N
// - C: [M x N],  ldc >= N (optional)
// - D: [M x N],  ldd >= N
// - Z: [M x N],  ldZ >= N (0이면 ldd 사용)
// - gY/gZ: [M x N], 각각 ldgY/ldgZ >= N
// - gA: [M x K], ldgA >= K; gB: [K x N], ldgB >= N; gC: [M x N], ldgC >= N
// ---------------------------------------------------------------------

// -------------------- Enums (ABI 고정: int 기반) --------------------
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

// -------------------- 기존 Forward 파라미터/런처 --------------------
struct GemmBiasActParams {
  // Shapes
  int M = 0, N = 0, K = 0;

  // Inputs
  const void* A = nullptr; int lda = 0;  // [M x K]
  const void* B = nullptr; int ldb = 0;  // [K x N]
  const void* C = nullptr; int ldc = 0;  // [M x N], optional

  // Output
  void*       D = nullptr; int ldd = 0;  // [M x N]

  // GEMM scalars
  float alpha = 1.0f;
  float beta  = 0.0f;

  // Epilogue
  const void* bias = nullptr;            // BiasKind에 따라 크기 달라짐 (아래 주석 참고)
  BiasKind bias_kind = BiasKind::None;
  ActKind  act       = ActKind::None;
  float    leaky_slope = 0.01f;          // LeakyReLU 전용
};

// 간단 테스트/스모크용 커널 런처(동일 시그니처 유지)
void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);

// 고수준 엔트리 (내부에서 cublasLt/커스텀 커널 선택)
void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s);

// -------------------- 확장 Forward: Z Stash/Workspace --------------------
struct GemmBiasActParamsEx {
  // Shapes
  int M = 0, N = 0, K = 0;

  // Inputs
  const void* A = nullptr; int lda = 0;  // [M x K], lda >= K
  const void* B = nullptr; int ldb = 0;  // [K x N], ldb >= N
  const void* C = nullptr; int ldc = 0;  // [M x N], optional

  // Output
  void* D = nullptr; int ldd = 0;        // [M x N], ldd >= N

  // GEMM scalars
  float alpha = 1.0f;
  float beta  = 0.0f;

  // Epilogue
  const void* bias = nullptr;            // BiasKind에 따라 크기 달라짐
  BiasKind bias_kind = BiasKind::None;

  ActKind act = ActKind::None;
  float   leaky_slope = 0.01f;

  // --- Pre-activation stash (선택) ---
  //   save_preact == 1 이면, 활성화 적용 전의 Z(= alpha*A*B + beta*C + bias)를 Z 버퍼에 기록
  void* Z   = nullptr;   // [M x N], optional
  int   ldZ = 0;         // 0 -> 내부에서 ldd로 대체

  int   save_preact = 0; // 1: Z에 저장, 0: 저장 안 함

  // --- cublasLt workspace (선택) ---
  void*  lt_workspace       = nullptr;   // capture-safe 외부 버퍼
  size_t lt_workspace_bytes = 0;         // 0이면 내부에서 Lt 없이 커스텀 경로 사용 가능
};

void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s);

// -------------------- Backward --------------------
struct GemmBiasActBwdParams {
  // Shapes
  int M = 0, N = 0, K = 0;

  // Forward inputs
  const void* A = nullptr; int lda = 0;  // [M x K]
  const void* B = nullptr; int ldb = 0;  // [K x N]
  const void* C = nullptr; int ldc = 0;  // [M x N], optional

  // Grad / stash
  const void* gY = nullptr; int ldgY = 0; // [M x N]
  const void* Z  = nullptr; int ldZ  = 0; // [M x N] (pre-activation, stash)

  // Output grads
  void* gA = nullptr; int ldgA = 0;      // [M x K]
  void* gB = nullptr; int ldgB = 0;      // [K x N]
  void* gC = nullptr; int ldgC = 0;      // [M x N], optional
  void* gBias = nullptr;                 // scalar/[M]/[N], bias_kind에 따름

  // Hyper-params
  float    alpha = 1.0f;
  float    beta  = 0.0f;
  BiasKind bias_kind   = BiasKind::None;
  ActKind  act         = ActKind::None;
  float    leaky_slope = 0.01f;

  // --- capture-safe scratch (dZ) ---
  // 외부에서 dZ(gZ) 임시 버퍼를 주면 내부 malloc/free 없이 사용 (그래프 캡처 안전)
  float* gZ_scratch = nullptr; // size >= M * N (row-major)
  int    ldgZ       = 0;       // 0이면 내부에서 N으로 취급/검증

  // --- cublasLt workspace (옵션) ---
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;
};

// Backward (확장 버전; 이름은 호환 유지)
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);

// -------------------- (선택) 유틸/검증 헬퍼 --------------------
// 구현 파일(api.hpp/launcher.cu)에서 사용. 헤더 인라인로 둬도 OK.
inline bool valid_ld_rowmajor(int rows, int cols, int ld) {
  if (rows <= 0 || cols <= 0) return false;
  if (ld == 0) return true;        // 0은 "내부 기본값 사용" 의미
  return ld >= cols;
}

inline int resolve_ld(int ld, int fallback_cols) {
  return (ld == 0) ? fallback_cols : ld;
}

// bias 버퍼 기대 크기 (float 기준). 타입이 다르면 호출부에서 스케일.
inline size_t expected_bias_elems(int M, int N, BiasKind k) {
  switch (k) {
    case BiasKind::None:   return 0;
    case BiasKind::Scalar: return 1;
    case BiasKind::PerM:   return (M > 0) ? static_cast<size_t>(M) : 0;
    case BiasKind::PerN:   return (N > 0) ? static_cast<size_t>(N) : 0;
    default:               return 0;
  }
}

// (옵션) Lt 워크스페이스 정렬 권장: 256B 이상 정렬
inline bool is_workspace_aligned(const void* p, size_t alignment = 256) {
  return (reinterpret_cast<uintptr_t>(p) % alignment) == 0;
}

} // namespace regemm
