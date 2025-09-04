#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <mutex>

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

// -------------------------- f32 스모크용 --------------------------
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

// 후처리 커널들 (폴백용)
__global__ void add_bias_fp16(
    __half* __restrict__ D,
    const float* __restrict__ bias,
    int M, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = M * N;
  for (int idx = tid; idx < size; idx += blockDim.x * gridDim.x) {
    int col = idx % N;
    float v = __half2float(D[idx]);
    v += bias[col];
    D[idx] = __float2half(v);
  }
}

__global__ void relu_only_fp16(__half* __restrict__ D, int M, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = M * N;
  for (int idx = tid; idx < size; idx += blockDim.x * gridDim.x) {
    float v = __half2float(D[idx]);
    if (v < 0.f) v = 0.f;
    D[idx] = __float2half(v);
  }
}

__global__ void add_bias_relu_fp16(
    __half* __restrict__ D,
    const float* __restrict__ bias,
    int M, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = M * N;
  for (int idx = tid; idx < size; idx += blockDim.x * gridDim.x) {
    int col = idx % N;
    float v = __half2float(D[idx]);
    v += bias[col];
    if (v < 0.f) v = 0.f;
    D[idx] = __float2half(v);
  }
}

// 핸들 캐싱
static cublasLtHandle_t get_lt_handle() {
  static cublasLtHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){ cublasLtCreate(&handle); });
  return handle;
}

extern "C" int ge2_launch_gemm_bias_act_tc_f16(
    const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (n < 4) return -1;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_opaque);

  const auto* p = reinterpret_cast<const GemmBiasActParams*>(bufs[n - 1]);
  if (!p) return -1;

  const __half* A = reinterpret_cast<const __half*>(bufs[0]);
  const __half* B = reinterpret_cast<const __half*>(bufs[1]);
  int idxD = p->has_bias ? 3 : 2;
  __half* D = reinterpret_cast<__half*>(bufs[idxD]);

  const float* bias_f32 = nullptr;
  if (p->has_bias) {
    if (n < 5) return -1;
    bias_f32 = reinterpret_cast<const float*>(bufs[2]);
  }

  const int64_t M = p->M, N = p->N, K = p->K;
  if (M <= 0 || N <= 0 || K <= 0) return -1;

  cublasLtHandle_t handle = get_lt_handle();
  if (!handle) return -2;

  // 람다: GEMM 실행 (try_epilogue_bias=true이면 Bias/ReluBias 에필로그 시도)
  auto run_matmul = [&](bool try_epilogue_bias, cublasStatus_t& out_st) -> bool {
    cublasLtMatmulDesc_t opDesc = nullptr;
    out_st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (out_st != CUBLAS_STATUS_SUCCESS) return false;

    // pointer mode host
    cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_HOST;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));

    cublasOperation_t transN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(transN));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(transN));

    if (try_epilogue_bias && p->has_bias) {
      cublasLtEpilogue_t epi =
          (p->act == 1) ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
      cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
      const void* biasDev = reinterpret_cast<const void*>(bias_f32);
      cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDev, sizeof(biasDev));
    }

    cublasLtMatrixLayout_t aDesc=nullptr, bDesc=nullptr, cDesc=nullptr, dDesc=nullptr;
    cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, M, K, K);
    cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, K, N, N);
    cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, M, N, N);
    cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_16F, M, N, N);
    set_row_major(aDesc); set_row_major(bDesc); set_row_major(cDesc); set_row_major(dDesc);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t max_ws = 0;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

    cublasLtMatmulHeuristicResult_t heur[8]; int returned = 0;
    out_st = cublasLtMatmulAlgoGetHeuristic(
        handle, opDesc, aDesc, bDesc, cDesc, dDesc, pref, 8, heur, &returned);

    void* ws = nullptr;
    if (out_st == CUBLAS_STATUS_SUCCESS && returned > 0 && heur[0].workspaceSize > 0) {
      cudaMalloc(&ws, heur[0].workspaceSize);
    }

    float alpha = 1.0f, beta = 0.0f;
    out_st = cublasLtMatmul(handle, opDesc,
                            &alpha,
                            A, aDesc,
                            B, bDesc,
                            &beta,
                            D, cDesc,
                            D, dDesc,
                            (returned>0? &heur[0].algo: nullptr),
                            ws, (ws ? heur[0].workspaceSize : 0),
                            stream);

    if (ws) cudaFree(ws);
    cublasLtMatmulPreferenceDestroy(pref);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (dDesc) cublasLtMatrixLayoutDestroy(dDesc);
    cublasLtMatmulDescDestroy(opDesc);
    return (out_st == CUBLAS_STATUS_SUCCESS);
  };

  // 1) 에필로그 시도
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  bool ok = run_matmul(p->has_bias, st);

  // 2) 실패 시 폴백: 에필로그 OFF + 후처리 커널
  if (!ok) {
    ok = run_matmul(false, st);
    if (!ok) return -2;

    int threads = 256;
    int blocks  = (int)((M * N + threads - 1) / threads);
    if (p->has_bias && p->act == 1) {
      add_bias_relu_fp16<<<blocks, threads, 0, stream>>>(D, bias_f32, (int)M, (int)N);
    } else if (p->has_bias && p->act == 0) {
      add_bias_fp16<<<blocks, threads, 0, stream>>>(D, bias_f32, (int)M, (int)N);
    } else if (!p->has_bias && p->act == 1) {
      relu_only_fp16<<<blocks, threads, 0, stream>>>(D, (int)M, (int)N);
    }
    if (cudaGetLastError() != cudaSuccess) return -2;
    return 0;
  }

  // 3) 에필로그 성공 + ReLU 단독 케이스
  if (!p->has_bias && p->act == 1) {
    int threads = 256;
    int blocks  = (int)((M * N + threads - 1) / threads);
    relu_only_fp16<<<blocks, threads, 0, stream>>>(D, (int)M, (int)N);
    if (cudaGetLastError() != cudaSuccess) return -2;
  }

  return 0;
}
