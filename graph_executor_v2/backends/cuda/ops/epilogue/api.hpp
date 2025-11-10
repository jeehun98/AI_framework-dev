#pragma once

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

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

// Dropout 모드
enum class DropoutMode : uint8_t {
  None = 0,
  MaskInput = 1,   // 외부 mask 제공 (0/1)
  Philox = 2       // Philox RNG로 내부 생성
};

// 에필로그 공통 속성 (+Dropout)
struct EpilogueAttrs {
  ActKind act         = ActKind::None;
  float   leaky_slope = 0.f;
  bool    save_z      = false;     // FWD에서만 의미

  // --- Dropout ---
  DropoutMode dmode   = DropoutMode::None;
  float       drop_p  = 0.f;       // 0<=p<1
  float       drop_scale = 1.f;    // 통상 1/(1-p)
  // Philox 모드용(옵션)
  uint64_t    rng_seed   = 0;
  uint64_t    rng_subseq = 0;
  uint64_t    rng_offset = 0;
};

// ---------- FWD ----------
struct EpilogueFwdParams {
  const float* X; int ldX;   // [M,N]
  const float* Bias;         // nullable
  BiasLayout   bias_layout;
  float*       Y; int ldY;   // [M,N]
  float*       Z; int ldZ;   // [M,N], save_z==true일 때만 의미
  int M, N;

  // Dropout mask 입력 (dmode==MaskInput일 때 사용)
  const uint8_t* DropMaskU8  = nullptr; // 0/1
  const float*   DropMaskF32 = nullptr; // 0.0/1.0
};

// ---------- BWD ----------
struct EpilogueBwdParams {
  const float* gY; int ldgY;   // [M,N]
  const float* Z;  int ldZ;    // [M,N] (pre-activation)
  float*       gZ; int ldgZ;   // [M,N]
  float*       gC; int ldgC;   // [M,N], optional (beta_for_gC != 0 일 때 의미)
  float        beta_for_gC;    // gC = beta * gZ
  float*       gBias;          // optional: reduce(gZ) into {Scalar|PerM|PerN}
  BiasLayout   bias_layout;
  int M, N;

  // Dropout mask 입력 (dmode==MaskInput일 때 사용, FWD과 동일 mask)
  const uint8_t* DropMaskU8  = nullptr;
  const float*   DropMaskF32 = nullptr;
};

// ---------- API (host dispatcher) ----------
ai::Status EpilogueFwdLaunch(const EpilogueFwdParams& p,
                             const EpilogueAttrs& a,
                             ai::StreamHandle stream);

ai::Status EpilogueBwdLaunch(const EpilogueBwdParams& p,
                             const EpilogueAttrs& a,
                             ai::StreamHandle stream);

} // namespace ai


// -------- CUDA kernels & host wrappers (정의는 kernels.cu) --------
void epilogue_get_launch_cfg(int M, int N, dim3& grid, dim3& block);

// device kernels
template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ,
         ai::DropoutMode DM>
__global__ void epilogue_fwd_kernel(const float* __restrict__ X, int ldX,
                                    const float* __restrict__ Bias,
                                    float* __restrict__ Y, int ldY,
                                    float* __restrict__ Z, int ldZ,
                                    int M, int N, float slope,
                                    float drop_p, float drop_scale,
                                    const void* __restrict__ DropMask,
                                    bool mask_is_float,
                                    uint64_t seed, uint64_t subseq, uint64_t offset);

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias,
         ai::DropoutMode DM>
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
    uint64_t seed, uint64_t subseq, uint64_t offset);

// host wrappers (런처에서 호출)
template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ, ai::DropoutMode DM>
void epilogue_fwd_launch(const float* X, int ldX,
                         const float* Bias,
                         float* Y, int ldY,
                         float* Z, int ldZ,
                         int M, int N, float slope,
                         float drop_p, float drop_scale,
                         const void* DropMask,
                         bool mask_is_float,
                         uint64_t seed, uint64_t subseq, uint64_t offset,
                         cudaStream_t s);

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias, ai::DropoutMode DM>
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
                         uint64_t seed, uint64_t subseq, uint64_t offset,
                         cudaStream_t s);
