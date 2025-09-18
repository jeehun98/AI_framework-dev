// src/regemm_backward.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <vector>

#include "regemm/api.h"
#include "regemm/activations.h"  // apply_act_grad_runtime()
#include "regemm/nvtx_shim.h"

namespace regemm {

// ======= 유틸: 에러 체크 =======
#ifndef REGEMM_CHECK
#define REGEMM_CHECK(stmt) do {                         \
  cudaError_t _e = (stmt);                              \
  if (_e != cudaSuccess) {                              \
    throw std::runtime_error(cudaGetErrorString(_e));   \
  }                                                     \
} while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(stmt) do {                         \
  cublasStatus_t _s = (stmt);                           \
  if (_s != CUBLAS_STATUS_SUCCESS) {                    \
    throw std::runtime_error("cuBLAS failure");         \
  }                                                     \
} while(0)
#endif

// === 디버그: BWD 헤더 값 확인용 ===
__global__ void debug_print_header_kernel(int* out_act,
                                          int* out_ldZ,
                                          int* out_ldgY,
                                          regemm::ActKind act,
                                          int ldZ, int ldgY)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out_act[0]  = static_cast<int>(act);
    out_ldZ[0]  = ldZ;
    out_ldgY[0] = ldgY;
  }
}

// ======= (호스트) 디버그: 헤더 블록 값 덤프 =======
static inline void dump_head_host(const char* tag,
                                  const float* Zp,  int ldZp,
                                  const float* gYp, int ldgYp,
                                  const float* gZp, int M, int N,
                                  cudaStream_t s)
{
  const int MM = (M < 4 ? M : 4);
  const int NN = (N < 6 ? N : 6);
  if (MM == 0 || NN == 0) return;

  std::vector<float> hZ(MM * NN), hY(MM * NN), hG(MM * NN);

  for (int m = 0; m < MM; ++m) {
    REGEMM_CHECK(cudaMemcpyAsync(&hZ[m * NN],  Zp  + m * ldZp,  sizeof(float) * NN, cudaMemcpyDeviceToHost, s));
    REGEMM_CHECK(cudaMemcpyAsync(&hY[m * NN],  gYp + m * ldgYp, sizeof(float) * NN, cudaMemcpyDeviceToHost, s));
    REGEMM_CHECK(cudaMemcpyAsync(&hG[m * NN],  gZp + m * N,     sizeof(float) * NN, cudaMemcpyDeviceToHost, s)); // gZ ld = N
  }
  REGEMM_CHECK(cudaStreamSynchronize(s));

  std::printf("[BWD dbg] %s (Z|gY|gZ) head:\n", tag);
  for (int m = 0; m < MM; ++m) {
    std::printf("  m=%d  Z:", m); for (int n = 0; n < NN; ++n) std::printf(" % .3f", hZ[m * NN + n]); std::printf("\n");
    std::printf("         Y:");   for (int n = 0; n < NN; ++n) std::printf(" % .3f", hY[m * NN + n]); std::printf("\n");
    std::printf("         G:");   for (int n = 0; n < NN; ++n) std::printf(" % .3f", hG[m * NN + n]); std::printf("\n");
  }
}

// ======= 3) SGEMM 래퍼 (row-major 편의) =======
// column-major 내부 cublas에 대해 C^T = (B op) * (A op) 로 호출
static inline cublasStatus_t sgemm_rm(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* alpha,
    const float* A, int lda_rm,
    const float* B, int ldb_rm,
    const float* beta,
    float* C, int ldc_rm)
{
  const cublasOperation_t opA_cm = transB ? CUBLAS_OP_T : CUBLAS_OP_N; // B op
  const cublasOperation_t opB_cm = transA ? CUBLAS_OP_T : CUBLAS_OP_N; // A op
  return cublasSgemm(
      h,
      opA_cm, opB_cm,
      /*m=*/N, /*n=*/M, /*k=*/K,
      alpha,
      /*A=*/B, /*lda=*/ldb_rm,
      /*B=*/A, /*ldb=*/lda_rm,
      beta,
      /*C=*/C, /*ldc=*/ldc_rm);
}

// ====================================================================
// ===================== NEW: BWD 에필로그 커널 ========================
// ====================================================================
//
//   gZ = gY ⊙ act'(Z)                      [항상]
//   (옵션) gC = beta * gZ                  [p.C && p.gC]
//   (옵션) gBias (Scalar/PerM/PerN) 축적   [p.gBias != nullptr]
//
// 메모리:
//   - 입력 gY(ldgY), Z(ldZ) : 주어진 stride 사용
//   - 출력 gZ : 내부 임시 버퍼 (contiguous, ld = N)  → GEMM에 바로 사용
//   - 출력 gC(ldgC) : 선택적
//
// 주의:
//   - Scalar bias는 atomic 누적이므로 사전에 0으로 초기화하는 편이 안전
//   - PerM/PerN도 atomicAdd 사용(간단 경로). 큰 사이즈에서 병목이면 2-phase 리덕션 권장.
//
template<ActKind AK, bool FUSE_GC>
__global__ void bwd_epilogue_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       // contiguous, ld = N
    int M, int N,
    float beta,                   // for gC
    float* __restrict__ gC, int ldgC, // nullable
    float* __restrict__ gBias,         // nullable
    BiasKind bk, float leaky_slope)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const float gy = gY[m * ldgY + n];
  const float z  = Z [m * ldZ  + n];

  // NOTE: activations.h에 템플릿 act_grad_apply<AK>()가 없다면,
  //       런타임 경로를 템플릿에 주입: apply_act_grad_runtime(z, gy, static_cast<ActKind>(AK), leaky_slope)
  const float gz = apply_act_grad_runtime(z, gy, static_cast<ActKind>(AK), leaky_slope);

  // 1) gZ 기록 (ld = N)
  gZ[m * N + n] = gz;

  // 2) gC (옵션)
  if constexpr (FUSE_GC) {
    if (gC) {
      gC[m * ldgC + n] = beta * gz;
    }
  }

  // 3) gBias (옵션) — 간단 경로: 원자 누적
  if (gBias) {
    if (bk == BiasKind::Scalar) {
      atomicAdd(gBias, gz);
    } else if (bk == BiasKind::PerM) {
      atomicAdd(&gBias[m], gz);
    } else if (bk == BiasKind::PerN) {
      atomicAdd(&gBias[n], gz);
    }
  }
}

// ====================================================================
// ============================ 메인 ================================
// ====================================================================
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s)
{
  NVTX_RANGE("regemm::bwd", 0xFFAA66);

  const int M = p.M, N = p.N, K = p.K;
  const int ldgY = p.ldgY;
  const int ldZ  = p.ldZ;

  // --- gZ 임시 버퍼 할당 (contiguous: ld = N) ---
  float* gZ = nullptr;
#if CUDART_VERSION >= 11020
  REGEMM_CHECK(cudaMallocAsync(&gZ, sizeof(float) * M * N, s));
#else
  REGEMM_CHECK(cudaMalloc(&gZ, sizeof(float) * M * N));
#endif

  // --- (디버그) 헤더 값 읽기 ---
  int *d_act=nullptr, *d_ldZ=nullptr, *d_ldgY=nullptr;
#if CUDART_VERSION >= 11020
  REGEMM_CHECK(cudaMallocAsync(&d_act,  sizeof(int), s));
  REGEMM_CHECK(cudaMallocAsync(&d_ldZ,  sizeof(int), s));
  REGEMM_CHECK(cudaMallocAsync(&d_ldgY, sizeof(int), s));
#else
  REGEMM_CHECK(cudaMalloc(&d_act,  sizeof(int)));
  REGEMM_CHECK(cudaMalloc(&d_ldZ,  sizeof(int)));
  REGEMM_CHECK(cudaMalloc(&d_ldgY, sizeof(int)));
#endif
  debug_print_header_kernel<<<1, 32, 0, s>>>(d_act, d_ldZ, d_ldgY, p.act, ldZ, ldgY);

  // --- NEW: BWD 에필로그 한 방에 처리 (gZ, [옵션]gC, [옵션]gBias) ---
  {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    const bool fuse_gC = (p.C && p.gC);

    // gBias는 atomicAdd 경로이므로 모드별 크기만큼 0으로 초기화 필요
    if (p.gBias) {
      size_t bytes = 0;
      if (p.bias_kind == BiasKind::Scalar)      bytes = sizeof(float);
      else if (p.bias_kind == BiasKind::PerM)   bytes = sizeof(float) * static_cast<size_t>(p.M);
      else if (p.bias_kind == BiasKind::PerN)   bytes = sizeof(float) * static_cast<size_t>(p.N);
      if (bytes) {
        REGEMM_CHECK(cudaMemsetAsync(p.gBias, 0, bytes, s));
      }
    }   

    switch (p.act) {
      case ActKind::ReLU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::ReLU, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::ReLU, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;

      case ActKind::LeakyReLU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::LeakyReLU, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::LeakyReLU, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;

      case ActKind::GELU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::GELU, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::GELU, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;

      case ActKind::Sigmoid:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::Sigmoid, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::Sigmoid, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;

      case ActKind::Tanh:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::Tanh, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::Tanh, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;

      case ActKind::None:
      default:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::None, true><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            p.beta, reinterpret_cast<float*>(p.gC), p.ldgC,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::None, false><<<grid, block, 0, s>>>(
            reinterpret_cast<const float*>(p.gY), ldgY,
            reinterpret_cast<const float*>(p.Z),  ldZ,
            gZ, M, N,
            0.f, nullptr, 0,
            reinterpret_cast<float*>(p.gBias), p.bias_kind, p.leaky_slope);
        break;
    }
  }

  // --- (디버그) 호스트로 값 복사 & 출력 ---
  int h_act=0, h_ldZ=0, h_ldgY=0;
  REGEMM_CHECK(cudaMemcpyAsync(&h_act,  d_act,  sizeof(int), cudaMemcpyDeviceToHost, s));
  REGEMM_CHECK(cudaMemcpyAsync(&h_ldZ,  d_ldZ,  sizeof(int), cudaMemcpyDeviceToHost, s));
  REGEMM_CHECK(cudaMemcpyAsync(&h_ldgY, d_ldgY, sizeof(int), cudaMemcpyDeviceToHost, s));
  REGEMM_CHECK(cudaStreamSynchronize(s));
  std::printf("[BWD dbg] act=%d (None=0,ReLU=1,Leaky=2,GELU=3,Sigmoid=4,Tanh=5) ldZ=%d ldgY=%d\n",
              h_act, h_ldZ, h_ldgY);

  // (옵션) Z, gY, gZ 헤더 영역 값 덤프
  dump_head_host("after gZ = gY * f'(Z)",
                 reinterpret_cast<const float*>(p.Z),  ldZ,
                 reinterpret_cast<const float*>(p.gY), ldgY,
                 gZ, M, N, s);

  // --- cuBLAS 준비 ---
  cublasHandle_t h = nullptr;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetStream(h, s));

  const float zero  = 0.f;
  const float alpha = p.alpha;

  // --- gA = alpha * gZ @ B^T  (M x K) = (M x N) @ (K x N)^T
  if (p.gA) {
    CUBLAS_CHECK(sgemm_rm(
      h,
      /*transA=*/false, /*transB=*/true,
      /*M=*/M, /*N=*/K, /*K=*/N,
      &alpha,
      /*A=*/gZ, /*lda=*/N,
      /*B=*/reinterpret_cast<const float*>(p.B), /*ldb=*/p.ldb,
      &zero,
      /*C=*/reinterpret_cast<float*>(p.gA), /*ldc=*/p.ldgA));
  }

  // --- gB = alpha * A^T @ gZ  (K x N) = (M x K)^T @ (M x N)
  if (p.gB) {
    CUBLAS_CHECK(sgemm_rm(
      h,
      /*transA=*/true, /*transB=*/false,
      /*M=*/K, /*N=*/N, /*K=*/M,
      &alpha,
      /*A=*/reinterpret_cast<const float*>(p.A), /*lda=*/p.lda,
      /*B=*/gZ, /*ldb=*/N,
      &zero,
      /*C=*/reinterpret_cast<float*>(p.gB), /*ldc=*/p.ldgB));
  }

  CUBLAS_CHECK(cublasDestroy(h));

  // --- 정리 ---
#if CUDART_VERSION >= 11020
  REGEMM_CHECK(cudaFreeAsync(gZ, s));
  REGEMM_CHECK(cudaFreeAsync(d_act,  s));
  REGEMM_CHECK(cudaFreeAsync(d_ldZ,  s));
  REGEMM_CHECK(cudaFreeAsync(d_ldgY, s));
#else
  REGEMM_CHECK(cudaFree(gZ));
  REGEMM_CHECK(cudaFree(d_act));
  REGEMM_CHECK(cudaFree(d_ldZ));
  REGEMM_CHECK(cudaFree(d_ldgY));
#endif
}

} // namespace regemm
