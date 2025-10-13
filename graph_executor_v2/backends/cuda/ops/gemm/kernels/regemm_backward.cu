// backends/cuda/ops/gemm/kernels/regemm_backward.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <unordered_map>

#include <regemm/config.h>
#include <regemm/bias.h>
#include <regemm/api.h>
#include <regemm/activations.h>
#include <regemm/nvtx_shim.h>

namespace regemm {

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

// ======= cublas 핸들: 디바이스별 캐시 (capture-safe) =======
static cublasHandle_t acquire_cublas_handle()
{
  static std::mutex mtx;
  static std::unordered_map<int, cublasHandle_t> handles;

  int dev = -1;
  REGEMM_CHECK(cudaGetDevice(&dev));

  {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = handles.find(dev);
    if (it != handles.end() && it->second) {
      return it->second;
    }
    // 최초 1회 생성: 반드시 캡처 밖(워밍업)에서 한 번 호출되도록 상위에서 보장
    cublasHandle_t h = nullptr;
    CUBLAS_CHECK(cublasCreate(&h));
    handles[dev] = h;
    return h;
  }
}

// row-major 편의 SGEMM 래퍼
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
  // row-major를 col-major로 부르는 트릭
  const cublasOperation_t opA_cm = transB ? CUBLAS_OP_T : CUBLAS_OP_N; // B op
  const cublasOperation_t opB_cm = transA ? CUBLAS_OP_T : CUBLAS_OP_N; // A op
  return cublasSgemm(
      h, opA_cm, opB_cm,
      /*m=*/N, /*n=*/M, /*k=*/K,
      alpha,
      /*A=*/B, /*lda=*/ldb_rm,
      /*B=*/A, /*ldb=*/lda_rm,
      beta,
      /*C=*/C, /*ldc=*/ldc_rm);
}

// ===================== BWD 에필로그 커널 =====================
// gZ = gY ⊙ act'(Z)
// (옵션) gC = beta * gZ
// (옵션) gBias 누적(Scalar/PerM/PerN)
template<ActKind AK, bool FUSE_GC>
__global__ void bwd_epilogue_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       // contiguous, ld = N
    int M, int N,
    float beta,                        // for gC
    float* __restrict__ gC, int ldgC,  // nullable
    float* __restrict__ gBias,         // nullable
    BiasKind bk, float leaky_slope)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const float gy = gY[m * ldgY + n];
  const float z  = Z [m * ldZ  + n];

  const float gz = apply_act_grad_runtime(z, gy, static_cast<ActKind>(AK), leaky_slope);

  // 1) gZ 기록 (ld = N)
  gZ[m * N + n] = gz;

  // 2) gC (옵션)
  if constexpr (FUSE_GC) {
    if (gC) gC[m * ldgC + n] = beta * gz;
  }

  // 3) gBias 누적 (간단 경로: atomicAdd)
  if (gBias) {
    if (bk == BiasKind::Scalar)      atomicAdd(gBias, gz);
    else if (bk == BiasKind::PerM)   atomicAdd(&gBias[m], gz);
    else if (bk == BiasKind::PerN)   atomicAdd(&gBias[n], gz);
  }
}

// ============================ 메인 ============================
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s)
{
  NVTX_RANGE("regemm::bwd", 0xFFAA66);

  const int M = p.M, N = p.N, K = p.K;
  const int ldgY = p.ldgY;
  const int ldZ  = p.ldZ;

  // 기본 가드
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("invalid dims");
  if (!p.gY || !p.Z) throw std::invalid_argument("gY/Z is null");
  if (ldgY < N || ldZ < N) throw std::invalid_argument("ldgY/ldZ < N");

  // -------- gZ scratch 준비 (캡처-세이프 우선) --------
  float* gZ = nullptr;
  bool need_free = false;

  if (p.gZ_scratch) {
    // 외부 제공 버퍼 사용 (ld == N 요구)
    if (p.ldgZ != 0 && p.ldgZ != N) {
      throw std::invalid_argument("gZ_scratch provided but ldgZ != N");
    }
    gZ = p.gZ_scratch;
  } else {
    // 내부 임시 할당 (비-캡처 경로 전용)
#if CUDART_VERSION >= 11020
    REGEMM_CHECK(cudaMallocAsync(&gZ, sizeof(float) * static_cast<size_t>(M) * N, s));
#else
    REGEMM_CHECK(cudaMalloc(&gZ, sizeof(float) * static_cast<size_t>(M) * N));
#endif
    need_free = true;
  }

  // -------- 에필로그 실행 (gZ, [옵션]gC, [옵션]gBias) --------
  {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    const bool fuse_gC = (p.C && p.gC);

    if (p.gBias) {
      size_t bytes = 0;
      if (p.bias_kind == BiasKind::Scalar)      bytes = sizeof(float);
      else if (p.bias_kind == BiasKind::PerM)   bytes = sizeof(float) * static_cast<size_t>(p.M);
      else if (p.bias_kind == BiasKind::PerN)   bytes = sizeof(float) * static_cast<size_t>(p.N);
      if (bytes) REGEMM_CHECK(cudaMemsetAsync(p.gBias, 0, bytes, s)); // capture-safe
    }

    switch (p.act) {
      case ActKind::ReLU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::ReLU, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::ReLU, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;

      case ActKind::LeakyReLU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::LeakyReLU, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::LeakyReLU, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;

      case ActKind::GELU:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::GELU, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::GELU, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;

      case ActKind::Sigmoid:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::Sigmoid, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::Sigmoid, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;

      case ActKind::Tanh:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::Tanh, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::Tanh, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;

      case ActKind::None:
      default:
        if (fuse_gC)
          bwd_epilogue_kernel<ActKind::None, true><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, p.beta, (float*)p.gC, p.ldgC, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        else
          bwd_epilogue_kernel<ActKind::None, false><<<grid, block, 0, s>>>(
            (const float*)p.gY, ldgY, (const float*)p.Z, ldZ,
            gZ, M, N, 0.f, nullptr, 0, (float*)p.gBias, p.bias_kind, p.leaky_slope);
        break;
    }
  }

  // -------- GEMMs (cuBLAS) --------
  // 🔴 기존: 매 호출 cublasCreate / Destroy → 캡처 무효화
  // ✅ 수정: 디바이스별 캐시 핸들 획득 + 스트림만 설정
  cublasHandle_t h = acquire_cublas_handle();
  CUBLAS_CHECK(cublasSetStream(h, s));

  const float zero  = 0.f;
  const float alpha = p.alpha;

  // gA = alpha * gZ @ B^T  (M x K) = (M x N) @ (K x N)^T
  if (p.gA) {
    CUBLAS_CHECK(sgemm_rm(
      h, /*transA=*/false, /*transB=*/true, M, K, N,
      &alpha,
      /*A=*/gZ, /*lda=*/N,
      /*B=*/(const float*)p.B, /*ldb=*/p.ldb,
      &zero,
      /*C=*/(float*)p.gA, /*ldc=*/p.ldgA));
  }

  // gB = alpha * A^T @ gZ  (K x N) = (M x K)^T @ (M x N)
  if (p.gB) {
    CUBLAS_CHECK(sgemm_rm(
      h, /*transA=*/true, /*transB=*/false, K, N, M,
      &alpha,
      /*A=*/(const float*)p.A, /*lda=*/p.lda,
      /*B=*/gZ, /*ldb=*/N,
      &zero,
      /*C=*/(float*)p.gB, /*ldc=*/p.ldgB));
  }

  // 🔵 핸들 파괴 금지 (프로세스 종료시 정리하거나 별도 shutdown API에서)
  // CUBLAS_CHECK(cublasDestroy(h)); // 제거

  // -------- gZ 해제 (내부 할당시에만) --------
  if (need_free) {
#if CUDART_VERSION >= 11020
    REGEMM_CHECK(cudaFreeAsync(gZ, s));
#else
    REGEMM_CHECK(cudaFree(gZ));
#endif
  }
}

} // namespace regemm
