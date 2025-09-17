#pragma once
#include <cstdint>

namespace regemm {

enum class ActKind : int {
  None = 0,
  ReLU = 1,
  LeakyReLU = 2,
  GELU = 3,
  Sigmoid = 4,
  Tanh = 5,
};

enum class BiasKind : int {
  None   = 0, // bias == nullptr
  Scalar = 1, // single scalar
  PerM   = 2, // one per row (M)
  PerN   = 3  // one per col (N)
};

struct GemmBiasActParams {
  // Matrix dims
  int M, N, K;

  // A: [M x K], lda >= K
  const void* A;
  int lda;

  // B: [K x N], ldb >= N
  const void* B;
  int ldb;

  // C: [M x N], optional, ldc >= N
  const void* C;
  int ldc;

  // D: [M x N], output, ldd >= N
  void* D;
  int ldd;

  // Scales
  float alpha; // multiplies accumulated A*B
  float beta;  // multiplies C

  // Bias
  const void* bias; // nullptr if none
  BiasKind bias_kind;

  // Activation
  ActKind act;
};

// Host launchers (implemented in launcher.cu)
void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);

void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s);

} // namespace regemm
