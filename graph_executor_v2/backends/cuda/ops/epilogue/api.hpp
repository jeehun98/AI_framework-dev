#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"   // Status, StreamHandle
#include "backends/cuda/ops/_common/shim/enums.hpp"     // ActKind, DropoutMode, BiasKind
#include "backends/cuda/ops/_common/shim/traits.hpp"    // BiasMode, to_bias_mode(...)

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Epilogue API types
// ------------------------------------------------------------

// 에필로그 공통 속성 (+Dropout)
struct EpilogueAttrs {
  ActKind     act         = ActKind::None;
  float       leaky_slope = 0.f;
  bool        save_z      = false;     // FWD에서만 의미

  // --- Dropout ---
  DropoutMode dmode       = DropoutMode::None;
  float       drop_p      = 0.f;       // 0<=p<1
  float       drop_scale  = 1.f;       // 통상 1/(1-p)
  // Philox 모드용(옵션)
  std::uint64_t rng_seed   = 0;
  std::uint64_t rng_subseq = 0;
  std::uint64_t rng_offset = 0;
};

// ---------- FWD ----------
struct EpilogueFwdParams {
  const float* X; int ldX;   // [M,N]
  const float* Bias;         // nullable
  BiasKind     bias_layout;  // (호환을 위해 이름 유지, 타입은 BiasKind)
  float*       Y; int ldY;   // [M,N]
  float*       Z; int ldZ;   // [M,N], save_z==true일 때만 의미
  int M, N;

  // Dropout mask 입력 (dmode==MaskInput일 때 사용)
  const std::uint8_t* DropMaskU8  = nullptr; // 0/1
  const float*        DropMaskF32 = nullptr; // 0.0/1.0
};

// ---------- BWD ----------
struct EpilogueBwdParams {
  const float* gY; int ldgY;   // [M,N]
  const float* Z;  int ldZ;    // [M,N] (pre-activation)
  float*       gZ; int ldgZ;   // [M,N]
  float*       gC; int ldgC;   // [M,N], optional (beta_for_gC != 0 일 때 의미)
  float        beta_for_gC;    // gC = beta * gZ
  float*       gBias;          // optional: reduce(gZ) into {Scalar|PerM|PerN}
  BiasKind     bias_layout;    // (호환명 유지)
  int M, N;

  // Dropout mask 입력 (dmode==MaskInput일 때 사용, FWD과 동일 mask)
  const std::uint8_t* DropMaskU8  = nullptr;
  const float*        DropMaskF32 = nullptr;
};

// ---------- API (host dispatcher) ----------
Status EpilogueFwdLaunch(const EpilogueFwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream);

Status EpilogueBwdLaunch(const EpilogueBwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream);

// ------------------------------------------------------------
// CUDA declarations (definitions in kernels.cu)
// ------------------------------------------------------------
void epilogue_get_launch_cfg(int M, int N, dim3& grid, dim3& block);

// device kernels
template<ActKind AK, BiasMode BM, bool SaveZ, DropoutMode DM>
__global__ void epilogue_fwd_kernel(const float* __restrict__ X, int ldX,
                                    const float* __restrict__ Bias,
                                    float* __restrict__ Y, int ldY,
                                    float* __restrict__ Z, int ldZ,
                                    int M, int N, float slope,
                                    float drop_p, float drop_scale,
                                    const void* __restrict__ DropMask,
                                    bool mask_is_float,
                                    std::uint64_t seed, std::uint64_t subseq, std::uint64_t offset);

template<ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias, DropoutMode DM>
__global__ void epilogue_bwd_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       int ldgZ,
    int M, int N,
    float beta_for_gC,
    float* __restrict__ gC, int ldgC,
    float* __restrict__ gBias,
    float leaky_slope,
    float drop_p, float drop_scale,
    const void* __restrict__ DropMask,
    bool mask_is_float,
    std::uint64_t seed, std::uint64_t subseq, std::uint64_t offset);

// host wrappers (런처에서 호출)
template<ActKind AK, BiasMode BM, bool SaveZ, DropoutMode DM>
void epilogue_fwd_launch(const float* X, int ldX,
                         const float* Bias,
                         float* Y, int ldY,
                         float* Z, int ldZ,
                         int M, int N, float slope,
                         float drop_p, float drop_scale,
                         const void* DropMask,
                         bool mask_is_float,
                         std::uint64_t seed, std::uint64_t subseq, std::uint64_t offset,
                         cudaStream_t s);

template<ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias, DropoutMode DM>
void epilogue_bwd_launch(const float* gY, int ldgY,
                         const float* Z,  int ldZ,
                         float* gZ,       int ldgZ,
                         int M, int N,
                         float beta_for_gC,
                         float* gC, int ldgC,
                         float* gBias,
                         float leaky_slope,
                         float drop_p, float drop_scale,
                         const void* DropMask,
                         bool mask_is_float,
                         std::uint64_t seed, std::uint64_t subseq, std::uint64_t offset,
                         cudaStream_t s);

} // namespace ai::cuda::shim
