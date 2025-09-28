#include <cuda_runtime.h>
#include <cstdint>

#include "regemm/config.h"
#include "regemm/api.h"
#include "regemm/activations.h"
#include "regemm/bias.h"
#include "regemm/nvtx_shim.h"

namespace regemm {

// ====================================================================
// 기존: Smoke (비타일, 소규모 행렬 최적)  — 이미 정식 스펙 준수
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

  // pre = alpha*(A@B) + beta*C + bias
  float pre = p.alpha * acc;
  if (p.beta != 0.f && C) {
    pre = fmaf(p.beta, C[m * p.ldc + n], pre);
  }
  pre += load_bias(p, m, n);

  // activation
  D[m * p.ldd + n] = apply_act_runtime(pre, p.act, p.leaky_slope);
}

void launch_gemm_bias_act_f32_smoke(const GemmBiasActParams& p, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);
  gemm_bias_act_f32_smoke<<<grid, block, 0, s>>>(p);
}

// ====================================================================
// 타일 파라미터(기존 유지)
// ====================================================================
constexpr int BM = REGEMM_TILE_M;
constexpr int BN = REGEMM_TILE_N;
constexpr int BK = REGEMM_TILE_K;

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
// Tiled kernel (고성능 경로) — 에필로그 수식 고정: pre=alpha*acc + beta*C + bias
// ====================================================================
template<int BM_, int BN_, int BK_, ActKind AK>
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
  for (int i = 0; i < THR_M; i++)
    #pragma unroll
    for (int j = 0; j < THR_N; j++)
      acc[i][j] = 0.f;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);

  auto load_A_tile = [&](int stage, int k0) {
    const int tid = ty * TDX + tx;
    const int elems = BM_ * BK_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      int r = e / BK_;
      int c = e % BK_;
      int gm = m0 + r;
      int gk = k0 + c;
      float v = 0.f;
      if (gm < p.M && gk < p.K) v = A[gm * p.lda + gk];
      As[stage][r][c] = v;
    }
  };

  auto load_B_tile = [&](int stage, int k0) {
    const int tid = ty * TDX + tx;
    const int elems = BK_ * BN_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      int r = e / BN_;
      int c = e % BN_;
      int gk = k0 + r;
      int gn = n0 + c;
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
      for (int i = 0; i < THR_M; i++) {
        int rm = tm0 + i;
        a_vec[i] = (rm < p.M) ? As[stage][rm - m0][kk] : 0.f;
      }

      float b_vec[THR_N];
      #pragma unroll
      for (int j = 0; j < THR_N; j++) {
        int cn = tn0 + j;
        b_vec[j] = (cn < p.N) ? Bs[stage][kk][cn - n0] : 0.f;
      }

      #pragma unroll
      for (int i = 0; i < THR_M; i++)
        #pragma unroll
        for (int j = 0; j < THR_N; j++)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
#if REGEMM_USE_DB
    stage ^= 1;
#endif
  }

  float bias_j[THR_N];
  #pragma unroll
  for (int j = 0; j < THR_N; j++) {
    int n = tn0 + j;
    bias_j[j] = load_bias(p, 0, (n < p.N ? n : 0));
  }

#if REGEMM_USE_VECIO
  constexpr int V = REGEMM_VEC_ALIGN_ELEMS;
  const uintptr_t Cptr = reinterpret_cast<uintptr_t>(C);
  const uintptr_t Dptr = reinterpret_cast<uintptr_t>(D);
  const bool base_ok   = ((tn0 % V) == 0) && ((THR_N % V) == 0);
  const bool d_align   = (Dptr % (V * sizeof(float)) == 0);
  const bool c_align   = (Cptr % (V * sizeof(float)) == 0);
  const bool vec_ok_store = base_ok && d_align && (p.ldd % V == 0);
  const bool vec_ok_loadC = base_ok && c_align && (p.ldc % V == 0);
#else
  const bool vec_ok_store = false;
  const bool vec_ok_loadC = false;
#endif

  const int ldc = p.ldc, ldd = p.ldd;

  #pragma unroll
  for (int i = 0; i < THR_M; i++) {
    int m = tm0 + i;
    if (m >= p.M) continue;

    float bias_m_cached = 0.f;
    if (p.bias && p.bias_kind == BiasKind::PerM) {
      bias_m_cached = load_bias(p, m, 0);
    }

    // --- 에필로그: pre = alpha*acc + beta*C + bias ---
    if (vec_ok_store) {
      #pragma unroll
      for (int j = 0; j < THR_N; j += 4) {
        int n = tn0 + j;
        float4 d4;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          int nn = n + t;
          int jj = j + t;
          if (nn < p.N) {
            float pre = p.alpha * acc[i][jj];
            if (p.beta != 0.f && C) {
              float cin = C[m * ldc + nn];
              pre = fmaf(p.beta, cin, pre);
            }
            if (p.bias) {
              if (p.bias_kind == BiasKind::PerN)      pre += bias_j[jj];
              else if (p.bias_kind == BiasKind::PerM) pre += bias_m_cached;
              else                                    pre += load_bias(p, m, nn);
            }
            (&d4.x)[t] = act_apply<AK>(pre, p.leaky_slope);
          }
        }
        if (n + 3 < p.N) {
          *reinterpret_cast<float4*>(&D[m * ldd + n]) = d4;
        } else {
          #pragma unroll
          for (int t = 0; t < 4; t++) {
            int nn = n + t;
            int jj = j + t;
            if (nn < p.N) D[m * ldd + nn] = (&d4.x)[t];
          }
        }
      }
    } else {
      #pragma unroll
      for (int j = 0; j < THR_N; j++) {
        int n = tn0 + j;
        if (n < p.N) {
          float pre = p.alpha * acc[i][j];
          if (p.beta != 0.f && C) {
            float cin = C[m * ldc + n];
            pre = fmaf(p.beta, cin, pre);
          }
          if (p.bias) {
            if (p.bias_kind == BiasKind::PerN)      pre += bias_j[j];
            else if (p.bias_kind == BiasKind::PerM) pre += bias_m_cached;
            else                                    pre += load_bias(p, m, n);
          }
          D[m * ldd + n] = act_apply<AK>(pre, p.leaky_slope);
        }
      }
    }
  }
}

void launch_gemm_bias_act_f32_tiled(const GemmBiasActParams& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);

  switch (p.act) {
    case ActKind::ReLU:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::ReLU><<<grid, block, 0, s>>>(p); break;
    case ActKind::LeakyReLU:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::LeakyReLU><<<grid, block, 0, s>>>(p); break;
    case ActKind::GELU:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::GELU><<<grid, block, 0, s>>>(p); break;
    case ActKind::Sigmoid:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::Sigmoid><<<grid, block, 0, s>>>(p); break;
    case ActKind::Tanh:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::Tanh><<<grid, block, 0, s>>>(p); break;
    case ActKind::None:
    default:
      gemm_bias_act_f32_tiled_kernel<BM, BN, BK, ActKind::None><<<grid, block, 0, s>>>(p); break;
  }
}

void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s) {
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke(p, s);
  else      launch_gemm_bias_act_f32_tiled(p, s);
}

// ====================================================================
// ======================  NEW: EX (Z Stash) 경로  =====================
// ====================================================================

// Smoke EX — 이미 정식 스펙 준수
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

  // pre = alpha*(A@B) + beta*C + bias
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

// Tiled EX — 에필로그 수식 고정 + Z stash
template<int BM_, int BN_, int BK_, ActKind AK>
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
  for (int i = 0; i < THR_M; i++)
    #pragma unroll
    for (int j = 0; j < THR_N; j++)
      acc[i][j] = 0.f;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__       D = reinterpret_cast<float*>(p.D);
  float* __restrict__       Z = reinterpret_cast<float*>(p.Z);
  const int ldZ = (p.ldZ == 0 ? p.ldd : p.ldZ);

  auto load_A_tile = [&](int stage, int k0) {
    const int tid = ty * TDX + tx;
    const int elems = BM_ * BK_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      int r = e / BK_;
      int c = e % BK_;
      int gm = m0 + r;
      int gk = k0 + c;
      float v = 0.f;
      if (gm < p.M && gk < p.K) v = A[gm * p.lda + gk];
      As[stage][r][c] = v;
    }
  };
  auto load_B_tile = [&](int stage, int k0) {
    const int tid = ty * TDX + tx;
    const int elems = BK_ * BN_;
    for (int e = tid; e < elems; e += (TDX * TDY)) {
      int r = e / BN_;
      int c = e % BN_;
      int gk = k0 + r;
      int gn = n0 + c;
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
      for (int i = 0; i < THR_M; i++) {
        int rm = tm0 + i;
        a_vec[i] = (rm < p.M) ? As[stage][rm - m0][kk] : 0.f;
      }

      float b_vec[THR_N];
      #pragma unroll
      for (int j = 0; j < THR_N; j++) {
        int cn = tn0 + j;
        b_vec[j] = (cn < p.N) ? Bs[stage][kk][cn - n0] : 0.f;
      }

      #pragma unroll
      for (int i = 0; i < THR_M; i++)
        #pragma unroll
        for (int j = 0; j < THR_N; j++)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
#if REGEMM_USE_DB
    stage ^= 1;
#endif
  }

  float bias_j[THR_N];
  #pragma unroll
  for (int j = 0; j < THR_N; j++) {
    int n = tn0 + j;
    bias_j[j] = load_bias(p, 0, (n < p.N ? n : 0));
  }

#if REGEMM_USE_VECIO
  constexpr int V = REGEMM_VEC_ALIGN_ELEMS;
  const uintptr_t Cptr = reinterpret_cast<uintptr_t>(C);
  const uintptr_t Dptr = reinterpret_cast<uintptr_t>(D);
  const bool base_ok   = ((tn0 % V) == 0) && ((THR_N % V) == 0);
  const bool d_align   = (Dptr % (V * sizeof(float)) == 0);
  const bool c_align   = (Cptr % (V * sizeof(float)) == 0);
  const bool vec_ok_store = base_ok && d_align && (p.ldd % V == 0);
  const bool vec_ok_loadC = base_ok && c_align && (p.ldc % V == 0);
#else
  const bool vec_ok_store = false;
  const bool vec_ok_loadC = false;
#endif

  const int ldc = p.ldc, ldd = p.ldd;

  #pragma unroll
  for (int i = 0; i < THR_M; i++) {
    int m = tm0 + i;
    if (m >= p.M) continue;

    float bias_m_cached = 0.f;
    if (p.bias && p.bias_kind == BiasKind::PerM) {
      bias_m_cached = load_bias(p, m, 0);
    }

    // --- 에필로그(+Z): pre = alpha*acc + beta*C + bias ---
    if (vec_ok_store) {
      #pragma unroll
      for (int j = 0; j < THR_N; j += 4) {
        int n = tn0 + j;
        float4 d4;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          int nn = n + t;
          int jj = j + t;
          if (nn < p.N) {
            float pre = p.alpha * acc[i][jj];
            if (p.beta != 0.f && C) {
              float cin = C[m * ldc + nn];
              pre = fmaf(p.beta, cin, pre);
            }
            if (p.bias) {
              if (p.bias_kind == BiasKind::PerN)      pre += bias_j[jj];
              else if (p.bias_kind == BiasKind::PerM) pre += bias_m_cached;
              else                                    pre += load_bias(p, m, nn);
            }
            if (p.save_preact && Z) { Z[m * ldZ + nn] = pre; }
            (&d4.x)[t] = act_apply<AK>(pre, p.leaky_slope);
          }
        }
        if (n + 3 < p.N) {
          *reinterpret_cast<float4*>(&D[m * ldd + n]) = d4;
        } else {
          #pragma unroll
          for (int t = 0; t < 4; t++) {
            int nn = n + t;
            if (nn < p.N) D[m * ldd + nn] = (&d4.x)[t];
          }
        }
      }
    } else {
      #pragma unroll
      for (int j = 0; j < THR_N; j++) {
        int n = tn0 + j;
        if (n < p.N) {
          float pre = p.alpha * acc[i][j];
          if (p.beta != 0.f && C) {
            float cin = C[m * ldc + n];
            pre = fmaf(p.beta, cin, pre);
          }
          if (p.bias) {
            if (p.bias_kind == BiasKind::PerN)      pre += bias_j[j];
            else if (p.bias_kind == BiasKind::PerM) pre += bias_m_cached;
            else                                    pre += load_bias(p, m, n);
          }
          if (p.save_preact && Z) { Z[m * ldZ + n] = pre; }
          D[m * ldd + n] = act_apply<AK>(pre, p.leaky_slope);
        }
      }
    }
  }
}

// EX 런처들
void launch_gemm_bias_act_f32_smoke_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);
  gemm_bias_act_f32_smoke_ex<<<grid, block, 0, s>>>(p);
}

void launch_gemm_bias_act_f32_tiled_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
  switch (p.act) {
    case ActKind::ReLU:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::ReLU><<<grid, block, 0, s>>>(p); break;
    case ActKind::LeakyReLU:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::LeakyReLU><<<grid, block, 0, s>>>(p); break;
    case ActKind::GELU:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::GELU><<<grid, block, 0, s>>>(p); break;
    case ActKind::Sigmoid:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::Sigmoid><<<grid, block, 0, s>>>(p); break;
    case ActKind::Tanh:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::Tanh><<<grid, block, 0, s>>>(p); break;
    case ActKind::None:
    default:
      gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, ActKind::None><<<grid, block, 0, s>>>(p); break;
  }
}

void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke_ex(p, s);
  else      launch_gemm_bias_act_f32_tiled_ex(p, s);
}

} // namespace regemm
