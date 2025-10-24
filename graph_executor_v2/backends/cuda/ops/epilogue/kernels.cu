#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <math_functions.h>

#include "backends/cuda/ops/gemm/detail/activations.h"
#include "backends/cuda/ops/gemm/detail/epilogue_adaptor.hpp"
#include "backends/cuda/ops/epilogue/api.hpp"

using regemm::BiasMode;

namespace {

// --- Philox 1 샘플(0/1 마스크)
__device__ __forceinline__
uint8_t philox_mask(uint64_t seed, uint64_t subseq, uint64_t offset,
                    uint64_t linear_idx, float p) {
  if (p <= 0.f) return 1;
  if (p >= 1.f) return 0;
  curandStatePhilox4_32_10_t st;
  // 각 요소를 다른 오프셋으로
  curand_init(seed, subseq, offset + linear_idx, &st);
  float r = curand_uniform(&st); // (0,1]
  return (r > p) ? 1 : 0;
}

// --- 활성화 전방
template<ai::ActKind AK>
__device__ __forceinline__ float apply_fwd(float x, float slope) {
  if constexpr (AK == ai::ActKind::None) {
    return x;
  } else if constexpr (AK == ai::ActKind::ReLU) {
    return x > 0.f ? x : 0.f;
  } else if constexpr (AK == ai::ActKind::LeakyReLU) {
    return x > 0.f ? x : slope * x;
  } else if constexpr (AK == ai::ActKind::Sigmoid) {
    // 간단한 안정성 보정
    float t = fminf(fmaxf(-x, -20.f), 20.f);
    return 1.f / (1.f + __expf(t));
  } else if constexpr (AK == ai::ActKind::Tanh) {
    return tanhf(x);
  } else if constexpr (AK == ai::ActKind::GELU) {
    // tanh 근사
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float u  = k0 * (x + k1 * x3);
    return 0.5f * x * (1.f + tanhf(u));
  } else {
    return x;
  }
}

// --- 활성화 역방
template<ai::ActKind AK>
__device__ __forceinline__ float apply_bwd(float z, float gy, float slope) {
  if constexpr (AK == ai::ActKind::None) {
    return gy;
  } else if constexpr (AK == ai::ActKind::ReLU) {
    return (z > 0.f ? 1.f : 0.f) * gy;
  } else if constexpr (AK == ai::ActKind::LeakyReLU) {
    return (z > 0.f ? 1.f : slope) * gy;
  } else if constexpr (AK == ai::ActKind::Sigmoid) {
    float t = fminf(fmaxf(-z, -20.f), 20.f);
    float s = 1.f / (1.f + __expf(t));
    return (s * (1.f - s)) * gy;
  } else if constexpr (AK == ai::ActKind::Tanh) {
    float th = tanhf(z);
    return (1.f - th * th) * gy;
  } else if constexpr (AK == ai::ActKind::GELU) {
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x  = z;
    float x2 = x * x;
    float u  = k0 * (x + k1 * x * x2);
    float th = tanhf(u);
    float sech2 = 1.f - th * th;
    float dudx = k0 * (1.f + 3.f * k1 * x2);
    float dgelu = 0.5f * (1.f + th) + 0.5f * x * sech2 * dudx;
    return dgelu * gy;
  } else {
    return gy;
  }
}

} // anon


// ===== 공통 launch cfg =====
void epilogue_get_launch_cfg(int M, int N, dim3& grid, dim3& block) {
  block = dim3(16,16);
  grid  = dim3((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
}


// ===== FWD kernel =====
template<ai::ActKind AK, BiasMode BM, bool SaveZ, ai::DropoutMode DM>
__global__ void epilogue_fwd_kernel(const float* __restrict__ X, int ldX,
                                    const float* __restrict__ Bias,
                                    float* __restrict__ Y, int ldY,
                                    float* __restrict__ Z, int ldZ,
                                    int M, int N, float slope,
                                    float drop_p, float drop_scale,
                                    const void* __restrict__ DropMask,
                                    bool mask_is_float,
                                    uint64_t seed, uint64_t subseq, uint64_t offset)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const int idxX = m * ldX + n;
  float x = X[idxX];

  if constexpr (BM == BiasMode::PerM)      { if (Bias) x += Bias[m]; }
  else if constexpr (BM == BiasMode::PerN) { if (Bias) x += Bias[n]; }
  else if constexpr (BM == BiasMode::Full) { if (Bias) x += *Bias; }

  if constexpr (SaveZ) { Z[m * ldZ + n] = x; }

  float y = apply_fwd<AK>(x, slope);

  if constexpr (DM != ai::DropoutMode::None) {
    uint8_t mk = 1;
    if constexpr (DM == ai::DropoutMode::MaskInput) {
      if (mask_is_float) {
        const float* F = static_cast<const float*>(DropMask);
        mk = (F[idxX] != 0.f);
      } else {
        const uint8_t* U = static_cast<const uint8_t*>(DropMask);
        mk = U[idxX];
      }
    } else { // Philox
      const uint64_t lin = static_cast<uint64_t>(m) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n);
      mk = philox_mask(seed, subseq, offset, lin, drop_p);
    }
    y = y * (mk ? drop_scale : 0.f);
  }

  Y[m * ldY + n] = y;
}


// ===== BWD kernel =====
template<ai::ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias, ai::DropoutMode DM>
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
    uint64_t seed, uint64_t subseq, uint64_t offset)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const int idxY = m * ldgY + n;
  const int idxZ = m * ldZ  + n;

  float gy = gY[idxY];

  // Dropout 역전파: gy *= (mask * scale)
  if constexpr (DM != ai::DropoutMode::None) {
    uint8_t mk = 1;
    if constexpr (DM == ai::DropoutMode::MaskInput) {
      if (mask_is_float) {
        const float* F = static_cast<const float*>(DropMask);
        mk = (F[idxY] != 0.f);
      } else {
        const uint8_t* U = static_cast<const uint8_t*>(DropMask);
        mk = U[idxY];
      }
    } else {
      const uint64_t lin = static_cast<uint64_t>(m) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n);
      mk = philox_mask(seed, subseq, offset, lin, drop_p);
    }
    gy = gy * (mk ? drop_scale : 0.f);
  }

  const float z  = Z[idxZ];
  const float gz = apply_bwd<AK>(z, gy, leaky_slope);
  gZ[m * ldgZ + n] = gz;

  if constexpr (FUSE_GC) {
    if (gC) gC[m * ldgC + n] = beta_for_gC * gz;
  }

  if constexpr (HasBias) {
    if constexpr (BM == BiasMode::PerM)      atomicAdd(&gBias[m], gz);
    else if constexpr (BM == BiasMode::PerN) atomicAdd(&gBias[n], gz);
    else if constexpr (BM == BiasMode::Full) atomicAdd(gBias, gz);
  }
}


// ===== host wrappers =====
template<ai::ActKind AK, BiasMode BM, bool SaveZ, ai::DropoutMode DM>
void epilogue_fwd_launch(const float* X, int ldX,
                         const float* Bias,
                         float* Y, int ldY,
                         float* Z, int ldZ,
                         int M, int N, float slope,
                         float drop_p, float drop_scale,
                         const void* DropMask,
                         bool mask_is_float,
                         uint64_t seed, uint64_t subseq, uint64_t offset,
                         cudaStream_t s)
{
  dim3 grid, block; epilogue_get_launch_cfg(M, N, grid, block);
  epilogue_fwd_kernel<AK, BM, SaveZ, DM>
      <<<grid, block, 0, s>>>(X, ldX, Bias, Y, ldY, Z, ldZ, M, N, slope,
                              drop_p, drop_scale, DropMask, mask_is_float,
                              seed, subseq, offset);
}

template<ai::ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias, ai::DropoutMode DM>
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
                         cudaStream_t s)
{
  dim3 grid, block; epilogue_get_launch_cfg(M, N, grid, block);
  epilogue_bwd_kernel<AK, FUSE_GC, BM, HasBias, DM>
      <<<grid, block, 0, s>>>(gY, ldgY, Z, ldZ, gZ, ldgZ, M, N,
                              beta_for_gC, gC, ldgC, gBias, leaky_slope,
                              drop_p, drop_scale, DropMask, mask_is_float,
                              seed, subseq, offset);
}


// ===== explicit instantiation (Act × Bias × SaveZ × DropoutMode) =====
// Mask dtype은 런타임 분기이므로 템플릿에서 제외 → 인스턴스 수 제한

#define INST_FWD(AK, BM, SAVEZ, DM) \
  template void epilogue_fwd_launch<ai::ActKind::AK, regemm::BiasMode::BM, SAVEZ, ai::DropoutMode::DM>( \
    const float*, int, const float*, float*, int, float*, int, int, int, float, \
    float, float, const void*, bool, uint64_t, uint64_t, uint64_t, cudaStream_t);

#define INST_BWD(AK, FUSE, BM, HASB, DM) \
  template void epilogue_bwd_launch<ai::ActKind::AK, FUSE, regemm::BiasMode::BM, HASB, ai::DropoutMode::DM>( \
    const float*, int, const float*, int, float*, int, int, int, float, float*, int, float*, float, \
    float, float, const void*, bool, uint64_t, uint64_t, uint64_t, cudaStream_t);

// 헬퍼
#define INST_FWD_ALL_DM(AK, BM, SAVEZ) \
  INST_FWD(AK, BM, SAVEZ, None) \
  INST_FWD(AK, BM, SAVEZ, MaskInput) \
  INST_FWD(AK, BM, SAVEZ, Philox)

#define INST_BWD_ALL_DM(AK, FUSE, BM, HASB) \
  INST_BWD(AK, FUSE, BM, HASB, None) \
  INST_BWD(AK, FUSE, BM, HASB, MaskInput) \
  INST_BWD(AK, FUSE, BM, HASB, Philox)

// 액티베이션별 인스턴스
#define INST_FOR_ACT(AK) \
  INST_FWD_ALL_DM(AK, None, false) INST_FWD_ALL_DM(AK, None, true) \
  INST_FWD_ALL_DM(AK, PerM, false) INST_FWD_ALL_DM(AK, PerM, true) \
  INST_FWD_ALL_DM(AK, PerN, false) INST_FWD_ALL_DM(AK, PerN, true) \
  INST_FWD_ALL_DM(AK, Full, false) INST_FWD_ALL_DM(AK, Full, true) \
  /* BWD: FUSE_GC=false/true × HasBias=false/true */ \
  INST_BWD_ALL_DM(AK, false, None, false) INST_BWD_ALL_DM(AK, false, None, true) \
  INST_BWD_ALL_DM(AK, false, PerM, false) INST_BWD_ALL_DM(AK, false, PerM, true) \
  INST_BWD_ALL_DM(AK, false, PerN, false) INST_BWD_ALL_DM(AK, false, PerN, true) \
  INST_BWD_ALL_DM(AK, false, Full, false) INST_BWD_ALL_DM(AK, false, Full, true) \
  INST_BWD_ALL_DM(AK, true,  None, false) INST_BWD_ALL_DM(AK, true,  None, true) \
  INST_BWD_ALL_DM(AK, true,  PerM, false) INST_BWD_ALL_DM(AK, true,  PerM, true) \
  INST_BWD_ALL_DM(AK, true,  PerN, false) INST_BWD_ALL_DM(AK, true,  PerN, true) \
  INST_BWD_ALL_DM(AK, true,  Full, false) INST_BWD_ALL_DM(AK, true,  Full, true)

INST_FOR_ACT(None)
INST_FOR_ACT(ReLU)
INST_FOR_ACT(LeakyReLU)
INST_FOR_ACT(GELU)
INST_FOR_ACT(Sigmoid)
INST_FOR_ACT(Tanh)

#undef INST_FOR_ACT
#undef INST_BWD_ALL_DM
#undef INST_FWD_ALL_DM
#undef INST_BWD
#undef INST_FWD
