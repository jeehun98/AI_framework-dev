// src/my_kernels.cu
#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>   // cublasCreate/cublasGetVersion
#include <mutex>
#include <cstdio>        // printf logs
#include <cstdlib>       // getenv, atoi
#include <cinttypes>

// --------- Optional NVTX (build with -DGE2_USE_NVTX=1) ---------
#if defined(GE2_USE_NVTX) && GE2_USE_NVTX
  #include <nvtx3/nvToolsExt.h>
  #define NVTX_PUSH(msg) nvtxRangePushA(msg)
  #define NVTX_POP()     nvtxRangePop()
#else
  #define NVTX_PUSH(msg) ((void)0)
  #define NVTX_POP()     ((void)0)
#endif
// ---------------------------------------------------------------

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

// 환경/버전 1회 로깅
static void log_env_once() {
  static std::once_flag once;
  std::call_once(once, [](){
    int cublas_ver = 0;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
      if (cublasGetVersion(h, &cublas_ver) != CUBLAS_STATUS_SUCCESS) cublas_ver = 0;
      cublasDestroy(h);
    }
    int dev = 0; 
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("[GE2] cublas=%d, gpu=%s, cc=%d%d\n",
                cublas_ver, prop.name, prop.major, prop.minor);
  });
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

// 후처리 커널들 (폴백/보조용)
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

// cuBLASLt 핸들 캐싱
static cublasLtHandle_t get_lt_handle() {
  static cublasLtHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      handle = nullptr;
    }
  });
  return handle;
}

// (옵션) 에필로그 enum 이름 출력용
static const char* epi_name(int v) {
  switch (v) {
    #ifdef CUBLASLT_EPILOGUE_DEFAULT
      case CUBLASLT_EPILOGUE_DEFAULT: return "DEFAULT";
    #endif
    #ifdef CUBLASLT_EPILOGUE_RELU
      case CUBLASLT_EPILOGUE_RELU: return "RELU";
    #endif
    #ifdef CUBLASLT_EPILOGUE_BIAS
      case CUBLASLT_EPILOGUE_BIAS: return "BIAS";
    #endif
    #ifdef CUBLASLT_EPILOGUE_RELU_BIAS
      case CUBLASLT_EPILOGUE_RELU_BIAS: return "RELU_BIAS";
    #endif
    #ifdef CUBLASLT_EPILOGUE_GELU
      case CUBLASLT_EPILOGUE_GELU: return "GELU";
    #endif
    #ifdef CUBLASLT_EPILOGUE_GELU_BIAS
      case CUBLASLT_EPILOGUE_GELU_BIAS: return "GELU_BIAS";
    #endif
    #ifdef CUBLASLT_EPILOGUE_BIAS_RELU
      case CUBLASLT_EPILOGUE_BIAS_RELU: return "BIAS_RELU";
    #endif
  }
  return "UNKNOWN";
}

extern "C" int ge2_launch_gemm_bias_act_tc_f16(
    const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (n < 4) return -1;
  log_env_once();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_opaque);

  const auto* p = reinterpret_cast<const GemmBiasActParams*>(bufs[n - 1]);
  if (!p) return -1;

  const __half* A = reinterpret_cast<const __half*>(bufs[0]); // [M,K]
  const __half* B = reinterpret_cast<const __half*>(bufs[1]); // [K,N]
  int idxD = p->has_bias ? 3 : 2;
  __half* D = reinterpret_cast<__half*>(bufs[idxD]);          // [M,N]

  const float* bias_f32 = nullptr;
  if (p->has_bias) {
    if (n < 5) return -1;
    bias_f32 = reinterpret_cast<const float*>(bufs[2]); // len=N
  }

  const int64_t M = p->M, N = p->N, K = p->K;
  if (M <= 0 || N <= 0 || K <= 0) return -1;

  cublasLtHandle_t handle = get_lt_handle();
  if (!handle) return -2;

  // ---- run_matmul: 에필로그 후보/여러 알고리즘 × WS 래더를 시도 ----
  auto run_matmul = [&](bool try_epilogue_bias, cublasStatus_t& out_st) -> bool {
    // Matmul desc
    cublasLtMatmulDesc_t opDesc = nullptr;
    out_st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (out_st != CUBLAS_STATUS_SUCCESS) return false;

    // host pointer mode (alpha/beta on host)
    cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_HOST;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));

    cublasOperation_t transN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(transN));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(transN));

    // 에필로그 속성 후보 적용 (가능한 것 중 하나라도 받아들이면 OK)
    auto set_epilogue_attrs = [&](cublasLtMatmulDesc_t desc, bool act1) -> bool {
      if (!try_epilogue_bias || !p->has_bias) return false;

      // 후보 목록(컴파일 환경이 지원하는 것만)
      cublasLtEpilogue_t candidates[6]; int nCand = 0;
      #ifdef CUBLASLT_EPILOGUE_RELU_BIAS
        if (act1) candidates[nCand++] = CUBLASLT_EPILOGUE_RELU_BIAS;
      #endif
      #ifdef CUBLASLT_EPILOGUE_BIAS_RELU
        if (act1) candidates[nCand++] = CUBLASLT_EPILOGUE_BIAS_RELU;
      #endif
      #ifdef CUBLASLT_EPILOGUE_BIAS
        candidates[nCand++] = CUBLASLT_EPILOGUE_BIAS;
      #endif
      // 필요 시 has_bias=0에서 RELU만 fuse하려면 아래 활성화
      // #ifdef CUBLASLT_EPILOGUE_RELU
      //   if (!p->has_bias && act1) candidates[nCand++] = CUBLASLT_EPILOGUE_RELU;
      // #endif

      if (nCand == 0) return false;

      // 에필로그 설정이 "성공"한 경우에만 bias 속성 지정
      for (int c = 0; c < nCand; ++c) {
        cublasStatus_t st_epi =
          cublasLtMatmulDescSetAttribute(
            desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &candidates[c], sizeof(candidates[c]));
        if (st_epi != CUBLAS_STATUS_SUCCESS) continue; // 이 후보 불가 → 다음 후보

        // bias dtype 시도 순서: FP32 → FP16 (환경 차 대비)
        cudaDataType_t bias_types[2] = { CUDA_R_32F, CUDA_R_16F };
        const void* biasDev = reinterpret_cast<const void*>(bias_f32);
        bool bias_ok = false;
        for (int bt = 0; bt < 2; ++bt) {
          #ifdef CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
            (void)cublasLtMatmulDescSetAttribute(
                desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_types[bt], sizeof(bias_types[bt]));
          #endif
          cublasStatus_t st_bp = cublasLtMatmulDescSetAttribute(
              desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDev, sizeof(biasDev));
          if (st_bp == CUBLAS_STATUS_SUCCESS) {
            std::printf("[GE2] epilogue accepted: %s  (bias_type=%s)\n",
                        epi_name((int)candidates[c]),
                        (bias_types[bt]==CUDA_R_32F?"fp32":"fp16"));
            bias_ok = true;
            break;
          }
        }
        if (bias_ok) return true;  // 에필로그 + bias 포인터 설정 완료
        // bias 속성까지는 거부 → 다음 후보 계속
      }

      // 어떤 후보도 완전하게(에필로그+바이어스) 세팅되지 않음
      return false;
    };

    bool have_epi = set_epilogue_attrs(opDesc, p->act == 1);
    if (try_epilogue_bias && !have_epi) {
      // 에필로그를 시도했지만 어떤 조합도 수용되지 않음 → 이 run 실패로 간주(곧바로 폴백 유도)
      std::printf("[GE2] epilogue not available on this config -> abort this try\n");
      cublasLtMatmulDescDestroy(opDesc);
      return false;
    }

    // Layouts (ROW major)
    cublasLtMatrixLayout_t aDesc=nullptr, bDesc=nullptr, cDesc=nullptr, dDesc=nullptr;
    if (cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, M, K, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, K, N, N) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, M, N, N) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&dDesc, CUDA_R_16F, M, N, N) != CUBLAS_STATUS_SUCCESS) {
      if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
      if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
      if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
      if (dDesc) cublasLtMatrixLayoutDestroy(dDesc);
      cublasLtMatmulDescDestroy(opDesc);
      return false;
    }
    set_row_major(aDesc); set_row_major(bDesc); set_row_major(cDesc); set_row_major(dDesc);
    std::printf("[GE2] desc/layouts ready (M=%lld N=%lld K=%lld)\n",
                (long long)M, (long long)N, (long long)K);

    // 여러 알고리즘 × WS 래더 시도(0 → 16MB → 64MB)
    auto try_with_ws_and_algo_list = [&](size_t ws_cap, cublasStatus_t& matmul_st) -> bool {
      cublasLtMatmulPreference_t pref = nullptr;
      cublasLtMatmulPreferenceCreate(&pref);
      cublasLtMatmulPreferenceSetAttribute(
          pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_cap, sizeof(ws_cap));

      cublasLtMatmulHeuristicResult_t heur[16]; int returned = 0;
      matmul_st = cublasLtMatmulAlgoGetHeuristic(
          handle, opDesc, aDesc, bDesc, cDesc, dDesc, pref, 16, heur, &returned);
      cublasLtMatmulPreferenceDestroy(pref);

      if (matmul_st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        std::printf("[GE2] heuristics none (ws=%zu)\n", ws_cap);
        return false;
      }

      for (int i = 0; i < returned; ++i) {
        void* ws = nullptr;
        if (heur[i].workspaceSize > 0) {
          if ((size_t)heur[i].workspaceSize > ws_cap) {
            std::printf("[GE2] skip algo #%d (needs ws=%zu > cap=%zu)\n",
                        i, (size_t)heur[i].workspaceSize, ws_cap);
            continue;
          }
          if (cudaMalloc(&ws, heur[i].workspaceSize) != cudaSuccess) ws = nullptr;
        }
        float alpha = 1.0f, beta = 0.0f;
        NVTX_PUSH("lt_gemm(run)");
        matmul_st = cublasLtMatmul(handle, opDesc,
                                   &alpha, A, aDesc, B, bDesc,
                                   &beta, D, cDesc, D, dDesc,
                                   &heur[i].algo,
                                   ws, (ws ? heur[i].workspaceSize : 0),
                                   stream);
        NVTX_POP();
        if (ws) cudaFree(ws);

        if (matmul_st == CUBLAS_STATUS_SUCCESS) {
          std::printf("[GE2] lt algo #%d ok (ws=%zu)\n", i, ws_cap);
          return true;
        }
        std::printf("[GE2] lt algo #%d failed (ws=%zu, st=%d)\n", i, ws_cap, (int)matmul_st);
      }
      return false;
    };

    cublasStatus_t matmul_st = CUBLAS_STATUS_SUCCESS;
    bool done = try_with_ws_and_algo_list(0, matmul_st) ||
                try_with_ws_and_algo_list(16ull<<20, matmul_st) ||
                try_with_ws_and_algo_list(64ull<<20, matmul_st);

    if (!done) std::printf("[GE2] matmul failed across all algos/ws (st=%d)\n", (int)matmul_st);

    // 정리
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (dDesc) cublasLtMatrixLayoutDestroy(dDesc);
    cublasLtMatmulDescDestroy(opDesc);
    out_st = matmul_st;
    return done;
  };

  // ----- 에필로그 시도/토글 -----
  bool fused_try = (p->has_bias != 0);
  if (const char* s = std::getenv("GE2_FORCE_NO_EPILOGUE")) {
    if (std::atoi(s) != 0) fused_try = false;
  }
  if (fused_try) {
    std::printf("[GE2] try epilogue: %s (M=%d N=%d K=%d)\n",
                (p->act==1?"RELU_BIAS":"BIAS"), p->M, p->N, p->K);
  } else {
    std::printf("[GE2] epilogue disabled (try=no-epi) (M=%d N=%d K=%d)\n",
                p->M, p->N, p->K);
  }

  // 1) 에필로그 시도
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  bool ok = run_matmul(fused_try, st);

  // 2) 실패 → 폴백: 에필로그 없이 GEMM + 후처리
  if (!ok) {
    std::printf("[GE2] epilogue failed -> fallback(post)\n");
    ok = run_matmul(false, st);
    if (!ok) return -2;

    int threads = 256;
    int blocks  = static_cast<int>((M * N + threads - 1) / threads);
    NVTX_PUSH("fallback_post");
    if (p->has_bias && p->act == 1) {
      add_bias_relu_fp16<<<blocks, threads, 0, stream>>>(D, bias_f32, (int)M, (int)N);
    } else if (p->has_bias && p->act == 0) {
      add_bias_fp16<<<blocks, threads, 0, stream>>>(D, bias_f32, (int)M, (int)N);
    } else if (!p->has_bias && p->act == 1) {
      relu_only_fp16<<<blocks, threads, 0, stream>>>(D, (int)M, (int)N);
    }
    NVTX_POP();
    if (cudaGetLastError() != cudaSuccess) return -2;
    return 0;
  }

  // 3) 에필로그 성공 + ReLU 단독 (has_bias=0, act=1) → post-activation
  if (!p->has_bias && p->act == 1) {
    std::printf("[GE2] no-bias + relu-only (post-activation)\n");
    int threads = 256;
    int blocks  = static_cast<int>((M * N + threads - 1) / threads);
    NVTX_PUSH("relu_only_post");
    relu_only_fp16<<<blocks, threads, 0, stream>>>(D, (int)M, (int)N);
    NVTX_POP();
    if (cudaGetLastError() != cudaSuccess) return -2;
  } else {
    std::printf("[GE2] epilogue fused OK\n");
  }

  return 0;
}
