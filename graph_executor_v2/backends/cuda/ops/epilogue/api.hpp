#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/gemm/detail/epilogue_adaptor.hpp" // regemm::BiasMode
#include <cstdint>

namespace ai {

// Bias 레이아웃(출력 [M,N] 기준)
enum class BiasLayout {
  None   = 0,  // 사용 안 함
  PerM   = 1,  // 행 기준: length=M
  PerN   = 2,  // 열 기준: length=N
  Scalar = 3   // 스칼라 1개
};

// 에필로그 공통 속성
struct EpilogueAttrs {
  ActKind act       = ActKind::None;
  float   leaky_slope = 0.f;
  bool    save_z      = false;  // FWD에서만 의미
};

// ---------- FWD ----------
struct EpilogueFwdParams {
  const float* X; int ldX;   // [M,N]
  const float* Bias;         // nullable
  BiasLayout   bias_layout;
  float*       Y; int ldY;   // [M,N]
  float*       Z; int ldZ;   // [M,N], save_z==true일 때만 의미 (nullable 허용 안 함)
  int M, N;
};

// ---------- BWD ----------
struct EpilogueBwdParams {
  const float* gY; int ldgY;   // [M,N]
  const float* Z;  int ldZ;    // [M,N] (pre-activation)
  float*       gZ; int ldgZ;   // [M,N] (contiguous 권장, ld=N)
  float*       gC; int ldgC;   // [M,N], optional (beta_for_gC != 0 일 때 의미)
  float        beta_for_gC;    // gC = beta * gZ
  float*       gBias;          // optional: reduce(gZ) into {Scalar|PerM|PerN}
  BiasLayout   bias_layout;
  int M, N;
};

// ---------- API ----------
ai::Status EpilogueFwdLaunch(const EpilogueFwdParams& p,
                             const EpilogueAttrs& a,
                             ai::StreamHandle stream);

ai::Status EpilogueBwdLaunch(const EpilogueBwdParams& p,
                             const EpilogueAttrs& a,
                             ai::StreamHandle stream);

} // namespace ai



// -------- CUDA kernels (선언만; 정의는 kernels.cu) --------
template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ>
__global__ void epilogue_fwd_kernel(const float* __restrict__ X, int ldX,
                                    const float* __restrict__ Bias,
                                    float* __restrict__ Y, int ldY,
                                    float* __restrict__ Z, int ldZ,
                                    int M, int N, float slope);

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias>
__global__ void epilogue_bwd_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       int ldgZ,
    int M, int N,
    float beta_for_gC,
    float* __restrict__ gC, int ldgC,
    float* __restrict__ gBias,
    float leaky_slope);

// -------- Host wrapper templates (여기를 런처가 호출) --------
void epilogue_get_launch_cfg(int M, int N, dim3& grid, dim3& block); // kernels.cu에서 정의

template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ>
void epilogue_fwd_launch(const float* X, int ldX,
                         const float* Bias,
                         float* Y, int ldY,
                         float* Z, int ldZ,
                         int M, int N, float slope,
                         cudaStream_t s);

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias>
void epilogue_bwd_launch(const float* gY, int ldgY,
                         const float* Z,  int ldZ,
                         float* gZ,       int ldgZ,
                         int M, int N,
                         float beta_for_gC,
                         float* gC, int ldgC,
                         float* gBias,
                         float leaky_slope,
                         cudaStream_t s);