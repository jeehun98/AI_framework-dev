// src/gemm_bias_act_tiled.cu
#include <cuda_runtime.h>
#include "regemm/api.h"
#include "regemm/activations.h"
#include "regemm/bias.h"
#include "regemm/config.h"

namespace regemm {

static __device__ __forceinline__ float apply_act(float x, ActKind k){
  if (k == ActKind::ReLU) return act_relu(x);
  return act_none(x);
}

constexpr int BM = REGEMM_TILE_M;
constexpr int BN = REGEMM_TILE_N;
constexpr int BK = REGEMM_TILE_K;

constexpr int TDX = REGEMM_BLOCK_TDX;
constexpr int TDY = REGEMM_BLOCK_TDY;

constexpr int THR_M = REGEMM_THREAD_TILE_M;
constexpr int THR_N = REGEMM_THREAD_TILE_N;

static_assert(TDX*THR_N == BN, "BN must equal TDX*THR_N");
static_assert(TDY*THR_M == BM, "BM must equal TDY*THR_M");

#if REGEMM_SMEM_PADK
  #define PADK 1
#else
  #define PADK 0
#endif

template<int BM_, int BN_, int BK_>
__global__ void gemm_bias_act_f32_tiled_kernel(GemmBiasActParams p) {
  // block origin
  const int m0 = blockIdx.y * BM_;
  const int n0 = blockIdx.x * BN_;

  // thread coords
  const int tx = threadIdx.x; // [0,TDX)
  const int ty = threadIdx.y; // [0,TDY)

  // micro-tile origin
  const int tm0 = m0 + ty * THR_M;
  const int tn0 = n0 + tx * THR_N;

  // shared memory tiles (double buffer)
#if REGEMM_USE_DB
  __shared__ float As[2][BM_][BK_+PADK];
  __shared__ float Bs[2][BK_][BN_+PADK];
#else
  __shared__ float As[1][BM_][BK_+PADK];
  __shared__ float Bs[1][BK_][BN_+PADK];
#endif

  // accumulators
  float acc[THR_M][THR_N];
  #pragma unroll
  for (int i=0;i<THR_M;i++)
    #pragma unroll
    for (int j=0;j<THR_N;j++)
      acc[i][j] = 0.f;

  const float* __restrict__ A = reinterpret_cast<const float*>(p.A);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.B);
  const float* __restrict__ C = reinterpret_cast<const float*>(p.C);
  float* __restrict__ D       = reinterpret_cast<float*>(p.D);

  // ====== loader helpers: cooperative load to SMEM ======
  auto load_A_tile = [&](int stage, int k0){
    // linearized cooperative load (scalar; 안전, 추후 float4 전환 가능)
    const int tid = ty*TDX + tx;
    const int elems = BM_*BK_;
    for (int e = tid; e < elems; e += (TDX*TDY)) {
      int r = e / BK_;
      int c = e % BK_;
      int gm = m0 + r;
      int gk = k0 + c;
      float v = 0.f;
      if (gm < p.M && gk < p.K) v = A[gm * p.lda + gk];
      As[stage][r][c] = v;
    }
  };
  auto load_B_tile = [&](int stage, int k0){
    const int tid = ty*TDX + tx;
    const int elems = BK_*BN_;
    for (int e = tid; e < elems; e += (TDX*TDY)) {
      int r = e / BN_;
      int c = e % BN_;
      int gk = k0 + r;
      int gn = n0 + c;
      float v = 0.f;
      if (gk < p.K && gn < p.N) v = B[gk * p.ldb + gn];
      Bs[stage][r][c] = v;
    }
  };

  // ====== prefetch first stage ======
  int stage = 0;
  if (p.K > 0) {
    load_A_tile(stage, 0);
    load_B_tile(stage, 0);
    __syncthreads();
  }

  // ====== K loop with (optional) double-buffering ======
  for (int k0 = 0; k0 < p.K; k0 += BK_) {

#if REGEMM_USE_DB
    // prefetch next while computing current
    int next = stage ^ 1;
    if (k0 + BK_ < p.K) {
      load_A_tile(next, k0 + BK_);
      load_B_tile(next, k0 + BK_);
    }
#endif

    // compute on current stage
    #pragma unroll
    for (int kk=0; kk<BK_; ++kk) {
      // a_vec: THR_M rows for this thread at column kk
      float a_vec[THR_M];
      #pragma unroll
      for (int i=0;i<THR_M;i++) {
        int rm = tm0 + i;
        a_vec[i] = (rm < p.M) ? As[stage][rm - m0][kk] : 0.f;
      }

      // b_vec: THR_N cols for this thread at row kk
      float b_vec[THR_N];
      #pragma unroll
      for (int j=0;j<THR_N;j++) {
        int cn = tn0 + j;
        b_vec[j] = (cn < p.N) ? Bs[stage][kk][cn - n0] : 0.f;
      }

      #pragma unroll
      for (int i=0;i<THR_M;i++)
        #pragma unroll
        for (int j=0;j<THR_N;j++)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
#if REGEMM_USE_DB
    stage = stage ^ 1; // swap
#endif
  }

  // ====== Epilogue (α/β/C + bias + act) & single store ======
  // per-N bias를 thread-local 레지스터에 캐시 (F: broadcast 최소화)
  float bias_j[THR_N];
  #pragma unroll
  for (int j=0;j<THR_N;j++) {
    int n = tn0 + j;
    bias_j[j] = load_bias(p, 0 /*unused*/, (n < p.N ? n : 0));
    // PerM/Scalar일 때도 load_bias가 알아서 처리(PerM은 m마다 다르므로 아래에서 더함)
  }

  // 벡터화 안전조건 (A): float4 store/load 조건 계산
#if REGEMM_USE_VECIO
  constexpr int V = REGEMM_VEC_ALIGN_ELEMS; // 4
  const bool vec_ok = (p.ldd % V == 0) && (p.ldc % V == 0) &&
                      ((tn0 % V) == 0) && (THR_N % V == 0);
#else
  const bool vec_ok = false;
#endif

  // C/D 포인터
  const int ldc = p.ldc, ldd = p.ldd;

  // 각 행(i)에 대해 THR_N 결과를 commit
  #pragma unroll
  for (int i=0;i<THR_M;i++) {
    int m = tm0 + i;
    if (m >= p.M) continue;

    // (옵션) β·C: vectorized read
    // j축 스트라이드 연속 → float4 로 읽기
    if (p.beta != 0.f && C) {
      if (vec_ok) {
        #pragma unroll
        for (int j=0; j<THR_N; j+=4) {
          int n = tn0 + j;
          if (n + 3 < p.N) {
            float4 c4 = *reinterpret_cast<const float4*>(&C[m*ldc + n]);
            acc[i][j+0] = fmaf(p.beta, c4.x, acc[i][j+0]);
            acc[i][j+1] = fmaf(p.beta, c4.y, acc[i][j+1]);
            acc[i][j+2] = fmaf(p.beta, c4.z, acc[i][j+2]);
            acc[i][j+3] = fmaf(p.beta, c4.w, acc[i][j+3]);
          } else {
            // 경계(잔여) 스칼라
            #pragma unroll
            for (int t=0;t<4;t++){
              int nn = n + t;
              if (nn < p.N) {
                float cin = C[m*ldc + nn];
                acc[i][nn - tn0] = fmaf(p.beta, cin, acc[i][nn - tn0]);
              }
            }
          }
        }
      } else {
        #pragma unroll
        for (int j=0;j<THR_N;j++) {
          int n = tn0 + j;
          if (n < p.N) {
            float cin = C[m*ldc + n];
            acc[i][j] = fmaf(p.beta, cin, acc[i][j]);
          }
        }
      }
    }

    // α 스케일 + bias + act
    #pragma unroll
    for (int j=0;j<THR_N;j++) {
      int n = tn0 + j;
      if (n < p.N) {
        float v = acc[i][j] * p.alpha;
        // PerN이면 bias_j[j], PerM이면 load_bias에서 m기준, Scalar도 처리
        v += (p.bias ? ((p.bias_kind==BiasKind::PerN) ? bias_j[j] : load_bias(p, m, n)) : 0.f);
        v = apply_act(v, p.act);
        acc[i][j] = v; // store 전 레지스터 보정
      }
    }

    // D store: vectorized
    if (vec_ok) {
      #pragma unroll
      for (int j=0; j<THR_N; j+=4) {
        int n = tn0 + j;
        if (n + 3 < p.N) {
          float4 d4;
          d4.x = acc[i][j+0];
          d4.y = acc[i][j+1];
          d4.z = acc[i][j+2];
          d4.w = acc[i][j+3];
          *reinterpret_cast<float4*>(&D[m*ldd + n]) = d4;
        } else {
          #pragma unroll
          for (int t=0;t<4;t++){
            int nn = n + t;
            if (nn < p.N) D[m*ldd + nn] = acc[i][j+t];
          }
        }
      }
    } else {
      #pragma unroll
      for (int j=0;j<THR_N;j++) {
        int n = tn0 + j;
        if (n < p.N) D[m*ldd + n] = acc[i][j];
      }
    }
  }
}

void launch_gemm_bias_act_f32_tiled(const GemmBiasActParams& p, cudaStream_t s){
  dim3 block(TDX, TDY);
  dim3 grid( (p.N + BN - 1)/BN, (p.M + BM - 1)/BM );
  gemm_bias_act_f32_tiled_kernel<BM,BN,BK><<<grid, block, 0, s>>>(p);
}

} // namespace regemm
