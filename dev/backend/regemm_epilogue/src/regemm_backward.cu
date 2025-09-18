#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>   // ✅ 추가: runtime_error 정의
#include <cstdio>  

#include "regemm/api.h"
#include "regemm/activations.h"  // apply_act_grad_runtime()
#include "regemm/nvtx_shim.h"

namespace regemm {

// ======= 유틸: 에러 체크(선택사항) =======
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

__global__ void scale_copy2d_kernel(
    float* __restrict__ out, int ldo,
    const float* __restrict__ in,  int ldi,
    float beta,
    int M, int N)
{
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M && n < N) out[m * ldo + n] = beta * in[m * ldi + n];
}


// ======= 1) gZ = gY ⊙ act'(Z) =======
__global__ void elemwise_act_backward_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       // contiguous, stride = N
    int M, int N,
    ActKind act, float leaky_slope)
{
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  float gy = gY[m * ldgY + n];
  float z  = Z [m * ldZ  + n];

  gZ[m * N + n] = apply_act_grad_runtime(z, gy, act, leaky_slope);
}

// ======= 2) gBias 리덕션 커널들 =======
// Scalar: gBias[0] = sum(gZ)
__global__ void bias_grad_scalar_kernel(const float* __restrict__ gZ, int size, float* gBias)
{
  // 간단 구현: block별 부분합 + atomicAdd
  __shared__ float sh[256];
  int tid = threadIdx.x;
  float sum = 0.f;

  for (int i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
    sum += gZ[i];
  }
  sh[tid] = sum;
  __syncthreads();

  // block reduce
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(gBias, sh[0]);
}

// PerM: gBias[m] = sum_n gZ[m, n]
__global__ void bias_grad_perM_kernel(const float* __restrict__ gZ, int M, int N, float* gBias)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  float s = 0.f;
  const float* row = gZ + m * N;
  for (int n = 0; n < N; ++n) s += row[n];
  gBias[m] = s;
}

// PerN: gBias[n] = sum_m gZ[m, n]
__global__ void bias_grad_perN_kernel(const float* __restrict__ gZ, int M, int N, float* gBias)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float s = 0.f;
  for (int m = 0; m < M; ++m) s += gZ[m * N + n];
  gBias[n] = s;
}

// ======= 3) SGEMM 래퍼 (row-major 편의) =======
// 주의: cublas는 column-major 가정이 기본. 여기서는 실용적으로
// row-major 버퍼를 그대로 사용하면서 opT 조합으로 계산한다.
// (이 방식은 앞서 제공한 sgemm 호출 컨벤션과 동일하게 맞춰둠)
// row-major 버퍼용 래퍼: C_rm = A_rm(@) B_rm
// 호출 시 인자 M,N,K, transA,transB, lda/ldb/ldc 모두 "row-major 관점" 그대로 넣으면 됨.
static inline cublasStatus_t sgemm_rm(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* alpha,
    const float* A, int lda_rm,   // row-major stride: cols
    const float* B, int ldb_rm,
    const float* beta,
    float* C, int ldc_rm)
{
  // column-major에서 계산: C^T = (B op) * (A op)
  const cublasOperation_t opA_cm = transB ? CUBLAS_OP_T : CUBLAS_OP_N; // <- B의 op
  const cublasOperation_t opB_cm = transA ? CUBLAS_OP_T : CUBLAS_OP_N; // <- A의 op

  // 주의: m=N, n=M, k=K 로 스왑
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


// ======= 4) 메인 엔트리: Backward(EX) =======
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s)
{
  NVTX_RANGE("regemm::bwd", 0xFFAA66);

  const int M = p.M, N = p.N, K = p.K;
  const int ldgY = p.ldgY;
  const int ldZ  = p.ldZ;

  // --- 4.1 gZ 임시 버퍼 할당 (contiguous: ld = N) ---
  float* gZ = nullptr;
#if CUDART_VERSION >= 11020
  REGEMM_CHECK(cudaMallocAsync(&gZ, sizeof(float) * M * N, s));
#else
  REGEMM_CHECK(cudaMalloc(&gZ, sizeof(float) * M * N));
#endif

  // --- 4.2 gZ = gY ⊙ act'(Z) ---
  {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    elemwise_act_backward_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const float*>(p.gY), ldgY,
        reinterpret_cast<const float*>(p.Z),  ldZ,
        gZ, M, N,
        p.act, p.leaky_slope);
  }

  // --- 4.3 cuBLAS 준비 ---
  cublasHandle_t h = nullptr;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetStream(h, s));

  const float one  = 1.f;
  const float zero = 0.f;
  const float alpha = p.alpha;

  // --- 4.4 gA = gZ @ B^T  (M x K) = (M x N) @ (K x N)^T
  if (p.gA) {
    CUBLAS_CHECK(sgemm_rm(
      h,
      /*transA=*/false, /*transB=*/true,
      /*M=*/M, /*N=*/K, /*K=*/N,
      &alpha,
      /*A=*/gZ, /*lda=*/N,
      /*B=*/reinterpret_cast<const float*>(p.B), /*ldb=*/N,
      &zero,
      /*C=*/reinterpret_cast<float*>(p.gA), /*ldc=*/K));
  }

  // --- 4.5 gB = A^T @ gZ  (K x N) = (M x K)^T @ (M x N)
  if (p.gB) {
    CUBLAS_CHECK(sgemm_rm(
      h,
      /*transA=*/true, /*transB=*/false,
      /*M=*/K, /*N=*/N, /*K=*/M,
      &alpha,
      /*A=*/reinterpret_cast<const float*>(p.A), /*lda=*/K,
      /*B=*/gZ, /*ldb=*/N,
      &zero,
      /*C=*/reinterpret_cast<float*>(p.gB), /*ldc=*/N));
  }

// --- 4.6 gC = beta * gZ (C 사용 시) ---
if (p.C && p.gC) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  scale_copy2d_kernel<<<grid, block, 0, s>>>(
      reinterpret_cast<float*>(p.gC), p.ldgC,
      /*in=*/gZ, /*ldi=*/N,
      /*beta=*/p.beta,
      M, N);
}


  // --- 4.7 gBias ---
  if (p.gBias != nullptr) {
    if (p.bias_kind == BiasKind::Scalar) {
      // gBias is scalar
      REGEMM_CHECK(cudaMemsetAsync(p.gBias, 0, sizeof(float), s));
      int threads = 256;
      int blocks = min(1024, (M * N + threads - 1) / threads);
      bias_grad_scalar_kernel<<<blocks, threads, 0, s>>>(gZ, M * N, reinterpret_cast<float*>(p.gBias));
    } else if (p.bias_kind == BiasKind::PerM) {
      int threads = 256;
      int blocks = (M + threads - 1) / threads;
      bias_grad_perM_kernel<<<blocks, threads, 0, s>>>(gZ, M, N, reinterpret_cast<float*>(p.gBias));
    } else if (p.bias_kind == BiasKind::PerN) {
      int threads = 256;
      int blocks = (N + threads - 1) / threads;
      bias_grad_perN_kernel<<<blocks, threads, 0, s>>>(gZ, M, N, reinterpret_cast<float*>(p.gBias));
    }
  }

  // --- 4.8 정리 ---
  CUBLAS_CHECK(cublasDestroy(h));
#if CUDART_VERSION >= 11020
  REGEMM_CHECK(cudaFreeAsync(gZ, s));
#else
  REGEMM_CHECK(cudaFree(gZ));
#endif
}

} // namespace regemm
