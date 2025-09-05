#include <cuda_runtime.h>
#include "regemm/api.h"
#include "regemm/activations.h"
#include "regemm/bias.h"

#ifndef REGEMM_TILE_M
#define REGEMM_TILE_M 128
#endif
#ifndef REGEMM_TILE_N
#define REGEMM_TILE_N 128
#endif
#ifndef REGEMM_TILE_K
#define REGEMM_TILE_K 16
#endif

// thread block: 16x16 = 256 threads
// each thread computes a 8x8 micro-tile => 16*8 =128 (M), 16*8 =128 (N)
static constexpr int BLK_M = REGEMM_TILE_M;
static constexpr int BLK_N = REGEMM_TILE_N;
static constexpr int BLK_K = REGEMM_TILE_K;
static constexpr int THR_M = 8;
static constexpr int THR_N = 8;

namespace regemm {

static __device__ __forceinline__ float apply_act(float x, ActKind k){
  if (k == ActKind::ReLU) return act_relu(x);
  return act_none(x);
}

template<int BM, int BN, int BK>
__global__ void gemm_bias_act_f32_tiled_kernel(GemmBiasActParams p) {
  // block origin
  int m0 = blockIdx.y * BM;
  int n0 = blockIdx.x * BN;

  // thread coordinates in block
  int tx = threadIdx.x; // [0,15]
  int ty = threadIdx.y; // [0,15]

  // micro-tile origin for this thread
  int tm0 = m0 + ty * THR_M;
  int tn0 = n0 + tx * THR_N;

  // shared memory tiles
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // accumulators
  float acc[THR_M][THR_N];
  #pragma unroll
  for (int i=0;i<THR_M;i++)
    #pragma unroll
    for (int j=0;j<THR_N;j++)
      acc[i][j] = 0.f;

  // K loop
  for (int k0 = 0; k0 < p.K; k0 += BK) {
    // load A tile: BM x BK
    #pragma unroll
    for (int i=0;i<THR_M;i++) {
      int m = tm0 + i;
      int k = k0 + tx; // reuse tx for vector-ish stride
      float v = 0.f;
      if (m < p.M && k < p.K) v = reinterpret_cast<const float*>(p.A)[m*p.lda + k];
      As[m - m0][k - k0] = v;
    }
    // load B tile: BK x BN
    #pragma unroll
    for (int j=0;j<THR_N;j++) {
      int n = tn0 + j;
      int k = k0 + ty; // reuse ty for other axis
      float v = 0.f;
      if (k < p.K && n < p.N) v = reinterpret_cast<const float*>(p.B)[k*p.ldb + n];
      Bs[k - k0][n - n0] = v;
    }

    __syncthreads();

    // compute micro tile
    #pragma unroll
    for (int kk=0; kk<BK; ++kk) {
      float a_vec[THR_M];
      float b_vec[THR_N];

      #pragma unroll
      for (int i=0;i<THR_M;i++) {
        int m = tm0 + i;
        a_vec[i] = (m < p.M) ? As[m - m0][kk] : 0.f;
      }
      #pragma unroll
      for (int j=0;j<THR_N;j++) {
        int n = tn0 + j;
        b_vec[j] = (n < p.N) ? Bs[kk][n - n0] : 0.f;
      }

      #pragma unroll
      for (int i=0;i<THR_M;i++)
        #pragma unroll
        for (int j=0;j<THR_N;j++)
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
    }

    __syncthreads();
  }

  // alpha/beta/C
  #pragma unroll
  for (int i=0;i<THR_M;i++) {
    int m = tm0 + i;
    #pragma unroll
    for (int j=0;j<THR_N;j++) {
      int n = tn0 + j;
      if (m < p.M && n < p.N) {
        float v = acc[i][j] * p.alpha;
        if (p.beta != 0.f && p.C) {
          float cin = reinterpret_cast<const float*>(p.C)[m*p.ldc + n];
          v = fmaf(p.beta, cin, v);
        }
        // bias
        v += load_bias(p, m, n);
        // act
        v = apply_act(v, p.act);
        // single store
        reinterpret_cast<float*>(p.D)[m*p.ldd + n] = v;
      }
    }
  }
}

void launch_gemm_bias_act_f32_tiled(const GemmBiasActParams& p, cudaStream_t s){
  dim3 block(16, 16);
  dim3 grid( (p.N + BLK_N - 1)/BLK_N, (p.M + BLK_M - 1)/BLK_M );
  gemm_bias_act_f32_tiled_kernel<BLK_M,BLK_N,BLK_K><<<grid, block, 0, s>>>(p);
}

} // namespace regemm
