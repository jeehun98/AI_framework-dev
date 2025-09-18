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

// ======= 내부 유틸 =======
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
  __shared__ float sh[256];
  int tid = threadIdx.x;
  float sum = 0.f;

  for (int i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
    sum += gZ[i];
  }
  sh[tid] = sum;
  __syncthreads();

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
// column-major 내부 cublas에 대해 C^T = (B op) * (A op)로 호출
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

  // --- 4.3 cuBLAS 준비 ---
  cublasHandle_t h = nullptr;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetStream(h, s));

  const float zero  = 0.f;
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
      REGEMM_CHECK(cudaMemsetAsync(p.gBias, 0, sizeof(float), s));
      int threads = 256;
      int blocks = (M * N + threads - 1) / threads;
      if (blocks > 1024) blocks = 1024;
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
