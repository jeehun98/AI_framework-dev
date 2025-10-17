// backends/cuda/ops/gemm/kernels/regemm_gemm_bias_act.cu
#include <cuda_runtime.h>
#include <cstdint>

#include "../detail/config.h"
#include "../detail/api.h"
#include "../detail/activations.h"
#include "../detail/bias.h"
#include "../detail/traits.hpp"
#include "../detail/nvtx_shim.h"

namespace regemm {

// ====================================================================
// 공통 타일/스레드 파라미터
// ====================================================================
constexpr int BM  = REGEMM_TILE_M;
constexpr int BN  = REGEMM_TILE_N;
constexpr int BK  = REGEMM_TILE_K;

constexpr int TDX = REGEMM_BLOCK_TDX;
constexpr int TDY = REGEMM_BLOCK_TDY;

constexpr int THR_M = REGEMM_THREAD_TILE_M;
constexpr int THR_N = REGEMM_THREAD_TILE_N;

static_assert(TDX * THR_N == BN, "BN must equal TDX*THR_N");
static_assert(TDY * THR_M == BM, "BM must equal TDY*THR_M");

#if REGEMM_SMEM_PADK
  #define PADK 1
#else
  #define PADK 0
#endif

#if REGEMM_SMEM_PADN
  #define PADN 1
#else
  #define PADN 0
#endif

// ====================================================================
// Smoke (소규모/비타일) — 런타임 분기 유지 (FWD 기본)
// ====================================================================
__global__ void gemm_bias_act_f32_smoke(GemmBiasActParams p) {
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= p.M || n >= p.N) return;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);

  float acc = 0.f;
  for (int k = 0; k < p.K; ++k) {
    float a = A[m * p.lda + k];
    float b = B[k * p.ldb + n];
    acc = fmaf(a, b, acc);
  }

  float pre = p.alpha * acc;
  if (p.beta != 0.f && C) {
    pre = fmaf(p.beta, C[m * p.ldc + n], pre);
  }
  pre += load_bias(p, m, n);

  D[m * p.ldd + n] = apply_act_runtime(pre, p.act, p.leaky_slope);
}

void launch_gemm_bias_act_f32_smoke(const GemmBiasActParams& p, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);
  gemm_bias_act_f32_smoke<<<grid, block, 0, s>>>(p);
}

// ====================================================================
// Tiled kernel (고성능 경로, FWD) — Epilogue 정책화(HasC/BiasMode)
//  * HasC: C(addend) 사용 여부 (FWD 일반적으로 false)
//  * BMmode: Bias 모드 (None/PerM/PerN/Full=Scalar)
//  * AK   : 활성화 종류
// ====================================================================
template<int BM_, int BN_, int BK_, ActKind AK, BiasMode BMmode, bool HasC>
__global__ void gemm_bias_act_f32_tiled_kernel(GemmBiasActParams p) {
  const int m0 = blockIdx.y * BM_;
  const int n0 = blockIdx.x * BN_;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tm0 = m0 + ty * THR_M;
  const int tn0 = n0 + tx * THR_N;

#if REGEMM_USE_DB
  __shared__ float As[2][BM_][BK_ + PADK];
  __shared__ float Bs[2][BK_][BN_ + PADN];
#else
  __shared__ float As[1][BM_][BK_ + PADK];
  __shared__ float Bs[1][BK_][BN_ + PADN];
#endif

  float acc[THR_M][THR_N];
  #pragma unroll
  for (int i = 0; i < THR_M; ++i)
    #pragma unroll
    for (int j = 0; j < THR_N; ++j)
      acc[i][j] = 0.f;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);

  auto load_A_tile = [&](int stage, int k0) {
    const int tid   = ty * TDX + tx;
    const int elems = BM_ * BK_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      const int r = e / BK_;
      const int c = e % BK_;
      const int gm = m0 + r;
      const int gk = k0 + c;
      float v = 0.f;
      if (gm < p.M && gk < p.K) v = A[gm * p.lda + gk];
      As[stage][r][c] = v;
    }
  };

  auto load_B_tile = [&](int stage, int k0) {
    const int tid   = ty * TDX + tx;
    const int elems = BK_ * BN_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      const int r = e / BN_;
      const int c = e % BN_;
      const int gk = k0 + r;
      const int gn = n0 + c;
      float v = 0.f;
      if (gk < p.K && gn < p.N) v = B[gk * p.ldb + gn];
      Bs[stage][r][c] = v;
    }
  };

  int stage = 0;
  if (p.K > 0) {
    load_A_tile(stage, 0);
    load_B_tile(stage, 0);
    __syncthreads();
  }

  for (int k0 = 0; k0 < p.K; k0 += BK_) {
#if REGEMM_USE_DB
    const int next = stage ^ 1;
    if (k0 + BK_ < p.K) {
      load_A_tile(next, k0 + BK_);
      load_B_tile(next, k0 + BK_);
    }
#endif

    #pragma unroll
    for (int kk = 0; kk < BK_; ++kk) {
      float a_vec[THR_M];
      #pragma unroll
      for (int i = 0; i < THR_M; ++i) {
        const int rm = tm0 + i;
        a_vec[i] = (rm < p.M) ? As[stage][rm - m0][kk] : 0.f;
      }

      float b_vec[THR_N];
      #pragma unroll
      for (int j = 0; j < THR_N; ++j) {
        const int cn = tn0 + j;
        b_vec[j] = (cn < p.N) ? Bs[stage][kk][cn - n0] : 0.f;
      }

      #pragma unroll
      for (int i = 0; i < THR_M; ++i)
        #pragma unroll
        for (int j = 0; j < THR_N; ++j)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
#if REGEMM_USE_DB
    stage ^= 1;
#endif
  }

  // PerN bias 프리패치
  float bias_j[THR_N];
  #pragma unroll
  for (int j = 0; j < THR_N; ++j) {
    const int n = tn0 + j;
    bias_j[j] = (n < p.N) ? load_bias(p, 0, n) : 0.f;
  }

  using EP = Epilogue<AK, BMmode, HasC, /*SaveZ*/false>;
  const int ldc = p.ldc, ldd = p.ldd;

  #pragma unroll
  for (int i = 0; i < THR_M; ++i) {
    const int m = tm0 + i;
    if (m >= p.M) continue;

    float bias_m_cached = 0.f;
    if constexpr (BMmode == BiasMode::PerM) {
      bias_m_cached = load_bias(p, m, 0);
    }

    #pragma unroll
    for (int j = 0; j < THR_N; ++j) {
      const int n = tn0 + j;
      if (n < p.N) {
        EP::apply(
          /*D*/ reinterpret_cast<float*>(p.D), ldd,
          /*C*/ reinterpret_cast<const float*>(p.C), ldc,
          /*Z*/ nullptr, 0,
          /*p*/ p,
          /*m*/ m, /*n*/ n,
          /*acc*/ acc[i][j],
          /*bias_j*/ (BMmode == BiasMode::PerN ? bias_j[j] : 0.f),
          /*bias_m*/ bias_m_cached
        );
      }
    }
  }
}

// ====================================================================
// EX (Z stash) — Smoke + Tiled(정책화: HasC/BiasMode/SaveZ)
// ====================================================================
__global__ void gemm_bias_act_f32_smoke_ex(GemmBiasActParamsEx p) {
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= p.M || n >= p.N) return;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);
  float* __restrict__       Z = reinterpret_cast<float*>(p.Z);

  const int ldZ = (p.ldZ == 0 ? p.ldd : p.ldZ);

  float acc = 0.f;
  for (int k = 0; k < p.K; ++k) {
    float a = A[m * p.lda + k];
    float b = B[k * p.ldb + n];
    acc = fmaf(a, b, acc);
  }

  float pre = p.alpha * acc;
  if (p.beta != 0.f && C) {
    pre = fmaf(p.beta, C[m * p.ldc + n], pre);
  }
  pre += load_bias(p, m, n);

  if (p.save_preact && Z) {
    Z[m * ldZ + n] = pre;
  }

  D[m * p.ldd + n] = apply_act_runtime(pre, p.act, p.leaky_slope);
}

template<int BM_, int BN_, int BK_, ActKind AK, BiasMode BMmode, bool HasC, bool SaveZ>
__global__ void gemm_bias_act_f32_tiled_kernel_ex(GemmBiasActParamsEx p) {
  const int m0 = blockIdx.y * BM_;
  const int n0 = blockIdx.x * BN_;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tm0 = m0 + ty * THR_M;
  const int tn0 = n0 + tx * THR_N;

#if REGEMM_USE_DB
  __shared__ float As[2][BM_][BK_ + PADK];
  __shared__ float Bs[2][BK_][BN_ + PADN];
#else
  __shared__ float As[1][BM_][BK_ + PADK];
  __shared__ float Bs[1][BK_][BN_ + PADN];
#endif

  float acc[THR_M][THR_N];
  #pragma unroll
  for (int i = 0; i < THR_M; ++i)
    #pragma unroll
    for (int j = 0; j < THR_N; ++j)
      acc[i][j] = 0.f;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);
  float* __restrict__       Z = reinterpret_cast<float*>(p.Z);
  const int ldZ = (p.ldZ == 0 ? p.ldd : p.ldZ);

  auto load_A_tile = [&](int stage, int k0) {
    const int tid   = ty * TDX + tx;
    const int elems = BM_ * BK_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      const int r = e / BK_;
      const int c = e % BK_;
      const int gm = m0 + r;
      const int gk = k0 + c;
      float v = 0.f;
      if (gm < p.M && gk < p.K) v = A[gm * p.lda + gk];
      As[stage][r][c] = v;
    }
  };
  auto load_B_tile = [&](int stage, int k0) {
    const int tid   = ty * TDX + tx;
    const int elems = BK_ * BN_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      const int r = e / BN_;
      const int c = e % BN_;
      const int gk = k0 + r;
      const int gn = n0 + c;
      float v = 0.f;
      if (gk < p.K && gn < p.N) v = B[gk * p.ldb + gn];
      Bs[stage][r][c] = v;
    }
  };

  int stage = 0;
  if (p.K > 0) {
    load_A_tile(stage, 0);
    load_B_tile(stage, 0);
    __syncthreads();
  }

  for (int k0 = 0; k0 < p.K; k0 += BK_) {
#if REGEMM_USE_DB
    const int next = stage ^ 1;
    if (k0 + BK_ < p.K) {
      load_A_tile(next, k0 + BK_);
      load_B_tile(next, k0 + BK_);
    }
#endif

    #pragma unroll
    for (int kk = 0; kk < BK_; ++kk) {
      float a_vec[THR_M];
      #pragma unroll
      for (int i = 0; i < THR_M; ++i) {
        const int rm = tm0 + i;
        a_vec[i] = (rm < p.M) ? As[stage][rm - m0][kk] : 0.f;
      }

      float b_vec[THR_N];
      #pragma unroll
      for (int j = 0; j < THR_N; ++j) {
        const int cn = tn0 + j;
        b_vec[j] = (cn < p.N) ? Bs[stage][kk][cn - n0] : 0.f;
      }

      #pragma unroll
      for (int i = 0; i < THR_M; ++i)
        #pragma unroll
        for (int j = 0; j < THR_N; ++j)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
#if REGEMM_USE_DB
    stage ^= 1;
#endif
  }

  // PerN bias 프리패치
  float bias_j[THR_N];
  #pragma unroll
  for (int j = 0; j < THR_N; ++j) {
    const int n = tn0 + j;
    bias_j[j] = (n < p.N) ? load_bias(p, 0, n) : 0.f;
  }

  using EP = Epilogue<AK, BMmode, HasC, SaveZ>;
  const int ldc = p.ldc, ldd = p.ldd;

  #pragma unroll
  for (int i = 0; i < THR_M; ++i) {
    const int m = tm0 + i;
    if (m >= p.M) continue;

    float bias_m_cached = 0.f;
    if constexpr (BMmode == BiasMode::PerM) {
      bias_m_cached = load_bias(p, m, 0);
    }

    #pragma unroll
    for (int j = 0; j < THR_N; ++j) {
      const int n = tn0 + j;
      if (n < p.N) {
        EP::apply(
          /*D*/ reinterpret_cast<float*>(p.D), ldd,
          /*C*/ reinterpret_cast<const float*>(p.C), ldc,
          /*Z*/ reinterpret_cast<float*>(p.Z), p.ldZ,
          /*p*/ p,
          /*m*/ m, /*n*/ n,
          /*acc*/ acc[i][j],
          /*bias_j*/ (BMmode == BiasMode::PerN ? bias_j[j] : 0.f),
          /*bias_m*/ bias_m_cached
        );
      }
    }
  }
}

// ====================================================================
// 런처들 (비-EX / EX)
// ====================================================================

// ---- 비-EX tiled 런처 ------------------------------------------------
template<ActKind AK, BiasMode BMmode, bool HasC>
static inline void launch_fwd_cfg(const GemmBiasActParams& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
  gemm_bias_act_f32_tiled_kernel<BM, BN, BK, AK, BMmode, HasC><<<grid, block, 0, s>>>(p);
}

template<ActKind AK, bool HasC>
static inline void launch_fwd_cfg_bm(const GemmBiasActParams& p, BiasMode bm, cudaStream_t s) {
  switch (bm) {
    case BiasMode::PerM: launch_fwd_cfg<AK, BiasMode::PerM, HasC>(p, s); break;
    case BiasMode::PerN: launch_fwd_cfg<AK, BiasMode::PerN, HasC>(p, s); break;
    case BiasMode::Full: launch_fwd_cfg<AK, BiasMode::Full, HasC>(p, s); break;
    case BiasMode::None:
    default:             launch_fwd_cfg<AK, BiasMode::None, HasC>(p, s); break;
  }
}

void launch_gemm_bias_act_f32_tiled(const GemmBiasActParams& p, cudaStream_t s) {
  const BiasMode bm = to_bias_mode(p.bias_kind);
  const bool hasC = (p.beta != 0.f && p.C != nullptr);

  switch (p.act) {
    case ActKind::ReLU:
      if (hasC) launch_fwd_cfg_bm<ActKind::ReLU,      true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::ReLU,      false>(p, bm, s);
      break;
    case ActKind::LeakyReLU:
      if (hasC) launch_fwd_cfg_bm<ActKind::LeakyReLU, true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::LeakyReLU, false>(p, bm, s);
      break;
    case ActKind::GELU:
      if (hasC) launch_fwd_cfg_bm<ActKind::GELU,      true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::GELU,      false>(p, bm, s);
      break;
    case ActKind::Sigmoid:
      if (hasC) launch_fwd_cfg_bm<ActKind::Sigmoid,   true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::Sigmoid,   false>(p, bm, s);
      break;
    case ActKind::Tanh:
      if (hasC) launch_fwd_cfg_bm<ActKind::Tanh,      true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::Tanh,      false>(p, bm, s);
      break;
    case ActKind::None:
    default:
      if (hasC) launch_fwd_cfg_bm<ActKind::None,      true >(p, bm, s);
      else      launch_fwd_cfg_bm<ActKind::None,      false>(p, bm, s);
      break;
  }
}

// ---- EX smoke 런처 ---------------------------------------------------
void launch_gemm_bias_act_f32_smoke_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);
  gemm_bias_act_f32_smoke_ex<<<grid, block, 0, s>>>(p);
}

// ---- EX tiled 런처 ---------------------------------------------------
template<ActKind AK, BiasMode BMmode, bool HasC, bool SaveZ>
static inline void launch_ex_cfg(const GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
  gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, AK, BMmode, HasC, SaveZ><<<grid, block, 0, s>>>(p);
}

template<ActKind AK, bool SaveZ>
static inline void launch_ex_cfg_bm(const GemmBiasActParamsEx& p, BiasMode bm, cudaStream_t s) {
  constexpr bool HasC = false; // EX FWD에서는 일반적으로 C 미사용
  switch (bm) {
    case BiasMode::PerM: launch_ex_cfg<AK, BiasMode::PerM, HasC, SaveZ>(p, s); break;
    case BiasMode::PerN: launch_ex_cfg<AK, BiasMode::PerN, HasC, SaveZ>(p, s); break;
    case BiasMode::Full: launch_ex_cfg<AK, BiasMode::Full, HasC, SaveZ>(p, s); break;
    case BiasMode::None:
    default:             launch_ex_cfg<AK, BiasMode::None, HasC, SaveZ>(p, s); break;
  }
}

void launch_gemm_bias_act_f32_tiled_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  const BiasMode bm = to_bias_mode(p.bias_kind);
  const bool saveZ = (p.save_preact != 0);

  switch (p.act) {
    case ActKind::ReLU:
      if (saveZ) launch_ex_cfg_bm<ActKind::ReLU,      true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::ReLU,      false>(p, bm, s);
      break;
    case ActKind::LeakyReLU:
      if (saveZ) launch_ex_cfg_bm<ActKind::LeakyReLU, true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::LeakyReLU, false>(p, bm, s);
      break;
    case ActKind::GELU:
      if (saveZ) launch_ex_cfg_bm<ActKind::GELU,      true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::GELU,      false>(p, bm, s);
      break;
    case ActKind::Sigmoid:
      if (saveZ) launch_ex_cfg_bm<ActKind::Sigmoid,   true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::Sigmoid,   false>(p, bm, s);
      break;
    case ActKind::Tanh:
      if (saveZ) launch_ex_cfg_bm<ActKind::Tanh,      true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::Tanh,      false>(p, bm, s);
      break;
    case ActKind::None:
    default:
      if (saveZ) launch_ex_cfg_bm<ActKind::None,      true >(p, bm, s);
      else       launch_ex_cfg_bm<ActKind::None,      false>(p, bm, s);
      break;
  }
}

// ====================================================================
// 최상위 FWD 엔트리 (호환 유지)
// ====================================================================
void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s) {
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke(p, s);
  else      launch_gemm_bias_act_f32_tiled(p, s);
}

void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke_ex(p, s);
  else      launch_gemm_bias_act_f32_tiled_ex(p, s);
}

// ====================================================================
// 명시적 인스턴스화 (링크 안정)
//  - 필요 활성화만 열어 바이너리 크기 제어 가능
// ====================================================================
namespace regemm {

// 템플릿 선언(정의는 위에 있음)
template<int,int,int,ActKind,BiasMode,bool>
__global__ void gemm_bias_act_f32_tiled_kernel(GemmBiasActParams);

template<int,int,int,ActKind,BiasMode,bool,bool>
__global__ void gemm_bias_act_f32_tiled_kernel_ex(GemmBiasActParamsEx);

// 현재 컴파일 유닛에서 사용하는 타일 값을 상수로 고정
constexpr int kBM = BM;
constexpr int kBN = BN;
constexpr int kBK = BK;

// ---------- FWD (non-EX) ----------
// HasC=false/true 모두 인스턴스화 (불필요하면 한쪽만 남기세요)
#define INSTANTIATE_FWD_FOR_ACT(AK) \
  /* BiasMode::None */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::None, false>(GemmBiasActParams); \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::None, true >(GemmBiasActParams); \
  /* BiasMode::PerM */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::PerM, false>(GemmBiasActParams); \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::PerM, true >(GemmBiasActParams); \
  /* BiasMode::PerN */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::PerN, false>(GemmBiasActParams); \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::PerN, true >(GemmBiasActParams); \
  /* BiasMode::Full(=Scalar) */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::Full, false>(GemmBiasActParams); \
  template __global__ void gemm_bias_act_f32_tiled_kernel<kBM,kBN,kBK, AK, BiasMode::Full, true >(GemmBiasActParams);

// 필요 활성화만 남기세요(전체 6종)
INSTANTIATE_FWD_FOR_ACT(ActKind::None)
INSTANTIATE_FWD_FOR_ACT(ActKind::ReLU)
INSTANTIATE_FWD_FOR_ACT(ActKind::LeakyReLU)
INSTANTIATE_FWD_FOR_ACT(ActKind::GELU)
INSTANTIATE_FWD_FOR_ACT(ActKind::Sigmoid)
INSTANTIATE_FWD_FOR_ACT(ActKind::Tanh)

#undef INSTANTIATE_FWD_FOR_ACT

// ---------- FWD EX (Z-stash) ----------
// HasC=false 고정(보통 FWD EX에서 C를 쓰지 않음). SaveZ=false/true 모두.
#define INSTANTIATE_EX_FOR_ACT(AK) \
  /* SaveZ=false */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::None, false, false>(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::PerM, false, false>(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::PerN, false, false>(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::Full, false, false>(GemmBiasActParamsEx); \
  /* SaveZ=true  */ \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::None, false, true >(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::PerM, false, true >(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::PerN, false, true >(GemmBiasActParamsEx); \
  template __global__ void gemm_bias_act_f32_tiled_kernel_ex<kBM,kBN,kBK, AK, BiasMode::Full, false, true >(GemmBiasActParamsEx);

// 필요 활성화만 남기세요(전체 6종)
INSTANTIATE_EX_FOR_ACT(ActKind::None)
INSTANTIATE_EX_FOR_ACT(ActKind::ReLU)
INSTANTIATE_EX_FOR_ACT(ActKind::LeakyReLU)
INSTANTIATE_EX_FOR_ACT(ActKind::GELU)
INSTANTIATE_EX_FOR_ACT(ActKind::Sigmoid)
INSTANTIATE_EX_FOR_ACT(ActKind::Tanh)

#undef INSTANTIATE_EX_FOR_ACT

} // namespace regemm

} // namespace regemm
