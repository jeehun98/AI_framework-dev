#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

// -------------------------- 공통 파라미터 블록 --------------------------
struct GemmBiasActParams {
  int M;         // A: MxK, B: KxN, D: MxN
  int N;
  int K;
  int has_bias;  // 0/1
  int act;       // 0:none, 1:ReLU
};

// Row-major 지정 헬퍼
static inline void set_row_major(cublasLtMatrixLayout_t lay) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayoutSetAttribute(
      lay, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
}

// -------------------------- f32 스모크용(유지) --------------------------
__global__ void gemm_bias_act_f32_kernel(
    float* __restrict__ D,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,  // nullable
    int M, int N, int K,
    int has_bias, int act) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc = fmaf(A[row*K + k], B[k*N + col], acc);
  if (has_bias && bias) acc += bias[col];
  if (act == 1 && acc < 0.f) acc = 0.f;
  D[row*N + col] = acc;
}

static int launch_gemm_bias_act_f32(
    const ge2_uintptr* bufs, int n, cudaStream_t stream) {
  if (n < 4) return -1;
  const auto* p = reinterpret_cast<const GemmBiasActParams*>(bufs[n - 1]);
  if (!p) return -1;

  const float* A = reinterpret_cast<const float*>(bufs[0]); // MxK
  const float* B = reinterpret_cast<const float*>(bufs[1]); // KxN
  const float* bias = nullptr;
  int idxD;
  if (p->has_bias) {
    if (n < 5) return -1;
    bias = reinterpret_cast<const float*>(bufs[2]); // N
    idxD = 3;
  } else {
    idxD = 2;
  }
  float* D = reinterpret_cast<float*>(bufs[idxD]); // MxN

  dim3 blk(16,16), grd((p->N+15)/16, (p->M+15)/16);
  gemm_bias_act_f32_kernel<<<grd, blk, 0, stream>>>(D, A, B, bias,
      p->M, p->N, p->K, p->has_bias, p->act);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int ge2_launch_gemm_bias_act_f32(
    const ge2_uintptr* bufs, int n, void* stream) {
  if (n < 4) return -1;
  return launch_gemm_bias_act_f32(bufs, n, reinterpret_cast<cudaStream_t>(stream));
}

// -------------------------- f16 + cuBLASLt ------------------------------
// 후처리 커널: bias(N) + ReLU (열 기준)
__global__ void add_bias_relu_fp16(
    __half* __restrict__ D,          // [M,N] row-major
    const float* __restrict__ bias,  // [N] fp32
    int M, int N,
    int has_bias, int act) {         // act: 0=none, 1=ReLU
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = M * N;
  for (int idx = tid; idx < size; idx += blockDim.x * gridDim.x) {
    int col = idx % N;
    float v = __half2float(D[idx]);
    if (has_bias && bias) v += bias[col];
    if (act == 1 && v < 0.f) v = 0.f;
    D[idx] = __float2half(v);
  }
}

extern "C" int ge2_launch_gemm_bias_act_tc_f16(
    const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (n < 4) return -1; // 최소 A,B,D,params
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_opaque);

  const auto* p = reinterpret_cast<const GemmBiasActParams*>(bufs[n - 1]);
  if (!p) return -1;

  // 입력/출력 (fp16, row-major)
  const __half* A = reinterpret_cast<const __half*>(bufs[0]); // MxK
  const __half* B = reinterpret_cast<const __half*>(bufs[1]); // KxN
  int idxD = p->has_bias ? 3 : 2;
  __half* D = reinterpret_cast<__half*>(bufs[idxD]);          // MxN

  // bias 포인터는 void*로 보관 (fp32 권장)
  const void* biasDev = nullptr;
  if (p->has_bias) {
    if (n < 5) return -1;
    biasDev = reinterpret_cast<const void*>(bufs[2]); // len=N, fp32
  }

  const int64_t M = p->M, N = p->N, K = p->K;
  if (M <= 0 || N <= 0 || K <= 0) return -1;

  // cuBLASLt 핸들/desc
  cublasLtHandle_t handle = nullptr;
  cublasStatus_t st = cublasLtCreate(&handle);
  if (st != CUBLAS_STATUS_SUCCESS) return -2;

  cublasLtMatmulDesc_t opDesc = nullptr;
  st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F); // FP32 accum
  if (st != CUBLAS_STATUS_SUCCESS) { cublasLtDestroy(handle); return -2; }

  cublasOperation_t transN = CUBLAS_OP_N; // Row-major N,N
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(transN));
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(transN));

  // (중요) 에필로그 미사용: bias/act는 후처리 커널로 처리

  cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr, dDesc = nullptr;
  cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, M, K, K); // ldA=K
  cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, K, N, N); // ldB=N
  cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, M, N, N); // ldC=N
  cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_16F, M, N, N); // ldD=N
  set_row_major(aDesc); set_row_major(bDesc); set_row_major(cDesc); set_row_major(dDesc);

  // Heuristic + workspace
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t max_ws = 0;
  cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

  cublasLtMatmulHeuristicResult_t heur[8]; int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(
      handle, opDesc, aDesc, bDesc, cDesc, dDesc, pref, 8, heur, &returned);

  void* ws = nullptr;
  if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
    // 0 WS에서 실패 → 16MB로 재시도
    cublasLtMatmulPreferenceDestroy(pref);
    size_t alt_ws = 16 << 20; // 16 MiB
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &alt_ws, sizeof(alt_ws));
    st = cublasLtMatmulAlgoGetHeuristic(
        handle, opDesc, aDesc, bDesc, cDesc, dDesc, pref, 8, heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
      cublasLtMatmulPreferenceDestroy(pref);
      cublasLtMatrixLayoutDestroy(aDesc); cublasLtMatrixLayoutDestroy(bDesc);
      cublasLtMatrixLayoutDestroy(cDesc); cublasLtMatrixLayoutDestroy(dDesc);
      cublasLtMatmulDescDestroy(opDesc);
      cublasLtDestroy(handle);
      return -2;
    }
    if (heur[0].workspaceSize > 0) {
      if (cudaMalloc(&ws, heur[0].workspaceSize) != cudaSuccess) ws = nullptr;
    }
  } else {
    if (heur[0].workspaceSize > 0) {
      if (cudaMalloc(&ws, heur[0].workspaceSize) != cudaSuccess) ws = nullptr;
    }
  }

  float alpha = 1.0f, beta = 0.0f;
  st = cublasLtMatmul(handle, opDesc,
                      &alpha,
                      A, aDesc,
                      B, bDesc,
                      &beta,
                      /*C in*/ D, cDesc,
                      /*D out*/ D, dDesc,
                      &heur[0].algo,
                      ws, (ws ? heur[0].workspaceSize : 0),
                      stream);

  if (ws) cudaFree(ws);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(aDesc); cublasLtMatrixLayoutDestroy(bDesc);
  cublasLtMatrixLayoutDestroy(cDesc); cublasLtMatrixLayoutDestroy(dDesc);
  cublasLtMatmulDescDestroy(opDesc);
  cublasLtDestroy(handle);

  if (st != CUBLAS_STATUS_SUCCESS) return -2;

  // 후처리: bias(N, fp32) + ReLU
  if (p->has_bias || p->act != 0) {
    const float* bias_f32 = reinterpret_cast<const float*>(biasDev);
    int threads = 256;
    int blocks  = (int)((M * N + threads - 1) / threads);
    add_bias_relu_fp16<<<blocks, threads, 0, stream>>>(D, bias_f32, (int)M, (int)N, p->has_bias, p->act);
    if (cudaGetLastError() != cudaSuccess) return -2;
  }

  return 0;
}
