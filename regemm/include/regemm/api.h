#pragma once
#include <cstdint>

namespace regemm {

enum class DType : int { F32 = 0 /* later: F16, BF16, TF32*/ };
enum class BiasKind : int { None=0, PerN=1, PerM=2, Scalar=3 };
enum class ActKind  : int { None=0, ReLU=1 /* later: GELU, SiLU... */ };

struct GemmBiasActParams {
  // Shapes: A[M,K], B[K,N], D[M,N]
  int M, N, K;

  // Scalars: D = Act( alpha * (A*B) + beta * C_in + Bias )
  float alpha{1.f}, beta{0.f};

  // Pointers (row-major 가정)
  const void* A{nullptr}; int lda{0}; // lda = K (row-major)
  const void* B{nullptr}; int ldb{0}; // ldb = N
  const void* C{nullptr}; int ldc{0}; // optional (beta!=0)
  void*       D{nullptr}; int ldd{0};

  // Bias/Act
  const void* bias{nullptr};
  BiasKind bias_kind{BiasKind::None};
  ActKind  act{ActKind::None};

  // Types / options
  DType dtype{DType::F32};
};

/// 런처(스모크: F32 전용)
int gemm_bias_act(const GemmBiasActParams& p, void* stream = nullptr);

} // namespace regemm
