// backends/cuda/ops/gemm/kernels/regemm_backward.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <unordered_map>

#include "backends/cuda/ops/gemm/api.hpp"                // REGEMM_* 매크로 + GemmBiasActBwdParams
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"    // 기본 shim 묶음
#include "backends/cuda/ops/_common/shim/activations.hpp"
#include "backends/cuda/ops/_common/shim/bias.hpp"
#include "backends/cuda/ops/_common/shim/traits.hpp"     // BiasMode, to_bias_mode
#include "backends/cuda/ops/_common/shim/ai_nvtx.hpp"

#include "backends/cuda/ops/gemm/kernels/config.hpp"


namespace ai::cuda::shim {

// ========== 에러 체크 유틸 ==========
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

// ======= cublas 핸들: 디바이스별 캐시 (capture-safe 생성은 상위에서 보장) =======
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
    // 최초 1회 생성: 반드시 그래프 캡처 바깥에서 워밍업 호출 필요
    cublasHandle_t h = nullptr;
    CUBLAS_CHECK(cublasCreate(&h));
    handles[dev] = h;
    return h;
  }
}

// row-major 편의 SGEMM 래퍼 (cublas는 column-major 기준)
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
  // row-major를 col-major로 호출: (A,B) 순서/전치 뒤집기
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

// ===================== BWD 에필로그 커널(정책화) =====================
// gZ = gY ⊙ act'(Z)
// (옵션) gC = beta * gZ
// (옵션) gBias 누적(Scalar/PerM/PerN)
template<ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias>
__global__ void bwd_epilogue_kernel(
    const float* __restrict__ gY, int ldgY,
    const float* __restrict__ Z,  int ldZ,
    float* __restrict__ gZ,       // contiguous, ld = N
    int M, int N,
    float beta,                        // for gC
    float* __restrict__ gC, int ldgC,  // nullable (FUSE_GC일 때만 의미)
    float* __restrict__ gBias,         // nullable (HasBias=false면 사용 안 함)
    float leaky_slope)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  const float gy = gY[m * ldgY + n];
  const float z  = Z [m * ldZ  + n];

  // 컴파일타임 활성화: 파생을 직접 곱해 오버헤드 줄임
  const float gz = gy * act_deriv<AK>(z, leaky_slope);

  // 1) gZ 기록 (ld = N)
  gZ[m * N + n] = gz;

  // 2) gC (옵션)
  if constexpr (FUSE_GC) {
    if (gC) gC[m * ldgC + n] = beta * gz;
  }

  // 3) gBias 누적 (정책화)
  if constexpr (HasBias) {
    if constexpr (BM == BiasMode::PerM) {
      atomicAdd(&gBias[m], gz);
    } else if constexpr (BM == BiasMode::PerN) {
      atomicAdd(&gBias[n], gz);
    } else if constexpr (BM == BiasMode::Full) { // Scalar
      atomicAdd(gBias, gz);
    }
  }
}

// 인스턴스 런처
template<ActKind AK, bool FUSE_GC, BiasMode BM, bool HasBias>
static inline void launch_bwd_epilogue_cfg(
    const GemmBiasActBwdParams& p, float* gZ, cudaStream_t s)
{
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);

  bwd_epilogue_kernel<AK, FUSE_GC, BM, HasBias><<<grid, block, 0, s>>>(
      reinterpret_cast<const float*>(p.gY), p.ldgY,
      reinterpret_cast<const float*>(p.Z),  p.ldZ,
      gZ, p.M, p.N,
      p.beta,
      reinterpret_cast<float*>(p.gC), p.ldgC,
      reinterpret_cast<float*>(p.gBias),
      p.leaky_slope);
}

// ============================ 메인 ============================
void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s)
{
  AI_NVTX_RANGE("regemm.bwd", nvtx::Color::Orange);

  const int M = p.M, N = p.N, K = p.K;
  const int ldgY = p.ldgY;
  const int ldZ  = p.ldZ;

  // 기본 가드
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("invalid dims");
  if (!p.gY || !p.Z)              throw std::invalid_argument("gY/Z is null");
  if (ldgY < N || ldZ < N)        throw std::invalid_argument("ldgY/ldZ < N");

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
#if CUDART_VERSION >= 11020
    REGEMM_CHECK(cudaMallocAsync(&gZ, sizeof(float) * static_cast<size_t>(M) * N, s));
#else
    REGEMM_CHECK(cudaMalloc(&gZ, sizeof(float) * static_cast<size_t>(M) * N));
#endif
    need_free = true;
  }

  // -------- gC 초기화 (요청됐지만 C가 없거나 beta==0) --------
  if (p.gC && (!p.C || p.beta == 0.f)) {
    AI_NVTX_RANGE("bwd.zero_gC", nvtx::Color::Cyan);
    const size_t bytes = sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(N);
    REGEMM_CHECK(cudaMemsetAsync(p.gC, 0, bytes, s));
  }

  // -------- gBias 0-fill (요청 시) --------
  if (p.gBias) {
    size_t bytes = 0;
    if (p.bias_kind == BiasKind::Scalar)      bytes = sizeof(float);
    else if (p.bias_kind == BiasKind::PerM)   bytes = sizeof(float) * static_cast<size_t>(p.M);
    else if (p.bias_kind == BiasKind::PerN)   bytes = sizeof(float) * static_cast<size_t>(p.N);
    if (bytes) REGEMM_CHECK(cudaMemsetAsync(p.gBias, 0, bytes, s)); // capture-safe
  }

  // -------- 에필로그 실행 (gZ, [옵션]gC, [옵션]gBias) --------
  {
    AI_NVTX_RANGE("bwd.epilogue", nvtx::Color::Green);

    const bool     fuse_gC = (p.C && p.gC && p.beta != 0.f);
    const BiasMode bm      = to_bias_mode(p.bias_kind);
    const bool     hasBias = (p.gBias != nullptr) && (bm != BiasMode::None);

    // 작은 디스패치 매크로
    #define DISPATCH_BIAS(AK_, FUSE_)                                      \
      switch (bm) {                                                        \
        case BiasMode::PerM:                                               \
          if (hasBias) launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::PerM, true >(p, gZ, s); \
          else         launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::PerM, false>(p, gZ, s); \
          break;                                                           \
        case BiasMode::PerN:                                               \
          if (hasBias) launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::PerN, true >(p, gZ, s); \
          else         launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::PerN, false>(p, gZ, s); \
          break;                                                           \
        case BiasMode::Full: /* Scalar */                                  \
          if (hasBias) launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::Full, true >(p, gZ, s); \
          else         launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::Full, false>(p, gZ, s); \
          break;                                                           \
        case BiasMode::None:                                               \
        default:                                                           \
          if (hasBias) launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::None, true >(p, gZ, s); \
          else         launch_bwd_epilogue_cfg<AK_, FUSE_, BiasMode::None, false>(p, gZ, s); \
          break;                                                           \
      }

    // ActKind × FUSE_GC 분기
    switch (p.act) {
      case ActKind::ReLU:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::ReLU,      true)  }
        else          { DISPATCH_BIAS(ActKind::ReLU,      false) }
        break;
      case ActKind::LeakyReLU:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::LeakyReLU, true)  }
        else          { DISPATCH_BIAS(ActKind::LeakyReLU, false) }
        break;
      case ActKind::GELU:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::GELU,      true)  }
        else          { DISPATCH_BIAS(ActKind::GELU,      false) }
        break;
      case ActKind::Sigmoid:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::Sigmoid,   true)  }
        else          { DISPATCH_BIAS(ActKind::Sigmoid,   false) }
        break;
      case ActKind::Tanh:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::Tanh,      true)  }
        else          { DISPATCH_BIAS(ActKind::Tanh,      false) }
        break;
      case ActKind::None:
      default:
        if (fuse_gC) { DISPATCH_BIAS(ActKind::None,      true)  }
        else          { DISPATCH_BIAS(ActKind::None,      false) }
        break;
    }

    #undef DISPATCH_BIAS
  }

  // -------- GEMMs (cuBLAS) --------
  AI_NVTX_RANGE("bwd.gemms", nvtx::Color::Red);

  // 디바이스별 캐시 핸들 획득 + 스트림 설정
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

  // -------- gZ 해제 (내부 할당시에만) --------
  if (need_free) {
#if CUDART_VERSION >= 11020
    REGEMM_CHECK(cudaFreeAsync(gZ, s));
#else
    REGEMM_CHECK(cudaFree(gZ));
#endif
  }
}

} // namespace ai::cuda::shim
