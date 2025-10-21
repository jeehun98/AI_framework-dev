#include <cuda_runtime.h>
#include "backends/cuda/ops/gemm/detail/activations.h"
#include "backends/cuda/ops/gemm/detail/epilogue_adaptor.hpp"
#include "backends/cuda/ops/epilogue/api.hpp"

using regemm::BiasMode;

#include <math_constants.h>  // CUDART_SQRT_HALF_PI 등 (선택)
#include <math_functions.h>

// ---- 로컬 활성화 전/후방 구현 (AK은 컴파일타임 상수) ----
template<ai::ActKind AK>
__device__ __forceinline__ float apply_fwd(float x, float slope) {
  if constexpr (AK == ai::ActKind::None) {
    return x;
  } else if constexpr (AK == ai::ActKind::ReLU) {
    return x > 0.f ? x : 0.f;
  } else if constexpr (AK == ai::ActKind::LeakyReLU) {
    return x > 0.f ? x : slope * x;
  } else if constexpr (AK == ai::ActKind::Sigmoid) {
    // 안정성 보정(간단): 큰 음수 클램프
    float t = fminf(fmaxf(-x, -20.f), 20.f);
    return 1.f / (1.f + __expf(t));
  } else if constexpr (AK == ai::ActKind::Tanh) {
    return tanhf(x);
  } else if constexpr (AK == ai::ActKind::GELU) {
    // tanh 근사: 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715x^3)))
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float u  = k0 * (x + k1 * x3);
    return 0.5f * x * (1.f + tanhf(u));
  } else {
    return x;
  }
}

template<ai::ActKind AK>
__device__ __forceinline__ float apply_bwd(float z, float gy, float slope) {
  if constexpr (AK == ai::ActKind::None) {
    return gy;
  } else if constexpr (AK == ai::ActKind::ReLU) {
    return (z > 0.f ? 1.f : 0.f) * gy;
  } else if constexpr (AK == ai::ActKind::LeakyReLU) {
    return (z > 0.f ? 1.f : slope) * gy;
  } else if constexpr (AK == ai::ActKind::Sigmoid) {
    // s = sigmoid(z), s' = s*(1-s)
    float t = fminf(fmaxf(-z, -20.f), 20.f);
    float s = 1.f / (1.f + __expf(t));
    return (s * (1.f - s)) * gy;
  } else if constexpr (AK == ai::ActKind::Tanh) {
    float th = tanhf(z);
    return (1.f - th * th) * gy;
  } else if constexpr (AK == ai::ActKind::GELU) {
    // tanh 근사의 도함수:
    // d/dx GELU(x) ≈ 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*k0*(1+3*k1*x^2)
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x  = z;
    float x2 = x * x;
    float u  = k0 * (x + k1 * x * x2);
    float th = tanhf(u);
    float sech2 = 1.f - th * th;            // sech^2(u)
    float dudx = k0 * (1.f + 3.f * k1 * x2);
    float dgelu = 0.5f * (1.f + th) + 0.5f * x * sech2 * dudx;
    return dgelu * gy;
  } else {
    return gy;
  }
}


// ===== kernels =====
template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ>
__global__ void epilogue_fwd_kernel(const float* __restrict__ X, int ldX,
                                    const float* __restrict__ Bias,
                                    float* __restrict__ Y, int ldY,
                                    float* __restrict__ Z, int ldZ,
                                    int M, int N, float slope)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  float x = X[m * ldX + n];
  if constexpr (BM == regemm::BiasMode::PerM)      { if (Bias) x += Bias[m]; }
  else if constexpr (BM == regemm::BiasMode::PerN) { if (Bias) x += Bias[n]; }
  else if constexpr (BM == regemm::BiasMode::Full) { if (Bias) x += *Bias; }

  if constexpr (SaveZ) { Z[m * ldZ + n] = x; }
  float y = apply_fwd<AK>(x, slope);
  Y[m * ldY + n] = y;
}

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias>
__global__ void epilogue_bwd_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       int ldgZ,
    int M, int N,
    float beta_for_gC,
    float* __restrict__ gC, int ldgC,
    float* __restrict__ gBias,
    float leaky_slope)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const float gy = gY[m * ldgY + n];
  const float z  = Z [m * ldZ  + n];
  const float gz = apply_bwd<AK>(z, gy, leaky_slope);
  gZ[m * ldgZ + n] = gz;

  if constexpr (FUSE_GC) { if (gC) gC[m * ldgC + n] = beta_for_gC * gz; }

  if constexpr (HasBias) {
    if constexpr (BM == regemm::BiasMode::PerM)      atomicAdd(&gBias[m], gz);
    else if constexpr (BM == regemm::BiasMode::PerN) atomicAdd(&gBias[n], gz);
    else if constexpr (BM == regemm::BiasMode::Full) atomicAdd(gBias, gz);
  }
}

// ===== common launch cfg =====
void epilogue_get_launch_cfg(int M, int N, dim3& grid, dim3& block) {
  block = dim3(16,16);
  grid  = dim3((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
}

// ===== host wrappers =====
template<ai::ActKind AK, regemm::BiasMode BM, bool SaveZ>
void epilogue_fwd_launch(const float* X, int ldX,
                         const float* Bias,
                         float* Y, int ldY,
                         float* Z, int ldZ,
                         int M, int N, float slope,
                         cudaStream_t s)
{
  dim3 grid, block; epilogue_get_launch_cfg(M, N, grid, block);
  epilogue_fwd_kernel<AK, BM, SaveZ><<<grid, block, 0, s>>>(X, ldX, Bias, Y, ldY, Z, ldZ, M, N, slope);
}

template<ai::ActKind AK, bool FUSE_GC, regemm::BiasMode BM, bool HasBias>
void epilogue_bwd_launch(const float* gY, int ldgY,
                         const float* Z,  int ldZ,
                         float* gZ,       int ldgZ,
                         int M, int N,
                         float beta_for_gC,
                         float* gC, int ldgC,
                         float* gBias,
                         float leaky_slope,
                         cudaStream_t s)
{
  dim3 grid, block; epilogue_get_launch_cfg(M, N, grid, block);
  epilogue_bwd_kernel<AK, FUSE_GC, BM, HasBias>
    <<<grid, block, 0, s>>>(gY, ldgY, Z, ldZ, gZ, ldgZ, M, N, beta_for_gC, gC, ldgC, gBias, leaky_slope);
}

// ===== explicit instantiation of HOST WRAPPERS (OK) =====
#define INST_FWD(AK, BM, SAVEZ) \
  template void epilogue_fwd_launch<ai::ActKind::AK, regemm::BiasMode::BM, SAVEZ>( \
    const float*, int, const float*, float*, int, float*, int, int, int, float, cudaStream_t);

#define INST_FWD_ACT(AK) \
  INST_FWD(AK, None, false) INST_FWD(AK, None, true) \
  INST_FWD(AK, PerM, false) INST_FWD(AK, PerM, true) \
  INST_FWD(AK, PerN, false) INST_FWD(AK, PerN, true) \
  INST_FWD(AK, Full, false) INST_FWD(AK, Full, true)

INST_FWD_ACT(None)
INST_FWD_ACT(ReLU)
INST_FWD_ACT(LeakyReLU)
INST_FWD_ACT(GELU)
INST_FWD_ACT(Sigmoid)
INST_FWD_ACT(Tanh)

#undef INST_FWD_ACT
#undef INST_FWD

#define INST_BWD(AK, FUSEGC, BM, HASB) \
  template void epilogue_bwd_launch<ai::ActKind::AK, FUSEGC, regemm::BiasMode::BM, HASB>( \
    const float*, int, const float*, int, float*, int, int, int, float, float*, int, float*, float, cudaStream_t);

#define INST_BWD_ACT_FUSE(AK, FUSE) \
  INST_BWD(AK, FUSE, None, false) INST_BWD(AK, FUSE, None, true) \
  INST_BWD(AK, FUSE, PerM, false) INST_BWD(AK, FUSE, PerM, true) \
  INST_BWD(AK, FUSE, PerN, false) INST_BWD(AK, FUSE, PerN, true) \
  INST_BWD(AK, FUSE, Full, false) INST_BWD(AK, FUSE, Full, true)

#define INST_BWD_ACT(AK) \
  INST_BWD_ACT_FUSE(AK, false) \
  INST_BWD_ACT_FUSE(AK, true)

INST_BWD_ACT(None)
INST_BWD_ACT(ReLU)
INST_BWD_ACT(LeakyReLU)
INST_BWD_ACT(GELU)
INST_BWD_ACT(Sigmoid)
INST_BWD_ACT(Tanh)

#undef INST_BWD_ACT
#undef INST_BWD_ACT_FUSE
#undef INST_BWD
