// run_graph_backward.cu (final, TF32 + strided-batched + no local handle)
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cublas_v2.h>

#include "run_graph.cuh"
#include "activation_ops.cuh"
#include "softmax_kernels.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss_kernels.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _st = (call); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS] %s:%d status=%d\n", __FILE__, __LINE__, (int)_st); \
    } \
} while(0)
#endif

// === 전역 cuBLAS 핸들 재사용 ===
static cublasHandle_t g_cublas = nullptr;
static void ensure_cublas() {
    if (!g_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
        // Ampere+ 성능 우선이면 주석 해제(전역 TF32)
        // cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    }
}

// 텐서/그래디언트 헬퍼
inline const float* get_tensor_ptr(
    const std::unordered_map<std::string, uintptr_t>& tensors,
    const std::string& id)
{
    auto it = tensors.find(id);
    if (it == tensors.end()) {
        fprintf(stderr, "[ERROR] Tensor ID '%s' not found (get_tensor_ptr)\n", id.c_str());
        return nullptr;
    }
    return reinterpret_cast<const float*>(it->second);
}

inline float* get_tensor_ptr_rw(
    std::unordered_map<std::string, uintptr_t>& tensors,
    const std::string& id)
{
    auto it = tensors.find(id);
    if (it == tensors.end()) {
        fprintf(stderr, "[ERROR] Tensor ID '%s' not found (get_tensor_ptr_rw)\n", id.c_str());
        return nullptr;
    }
    return reinterpret_cast<float*>(it->second);
}

inline const float* get_grad_ptr(
    const std::unordered_map<std::string, uintptr_t>& grads,
    const std::string& id)
{
    auto it = grads.find(id);
    if (it == grads.end()) {
        fprintf(stderr, "[ERROR] Grad ID '%s' not found (get_grad_ptr)\n", id.c_str());
        return nullptr;
    }
    return reinterpret_cast<const float*>(it->second);
}

inline float* ensure_grad(std::unordered_map<std::string, uintptr_t>& grads,
                          const std::unordered_map<std::string, Shape>& shapes,
                          const std::string& id,
                          const Shape& shape,
                          int batch_size)
{
    auto it = grads.find(id);
    if (it != grads.end()) {
        return reinterpret_cast<float*>(it->second);
    }
    size_t elems = static_cast<size_t>(batch_size) * shape.rows * shape.cols;
    float* dptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dptr, elems * sizeof(float)));
    grads[id] = reinterpret_cast<uintptr_t>(dptr);
    return dptr;
}

// 활성화 매핑
static inline int map_act_type(int op_type) {
    switch (op_type) {
        case SIGMOID:    return ACT_SIGMOID;
        case RELU:       return ACT_RELU;
        case TANH:       return ACT_TANH;
        case LEAKY_RELU: return ACT_LEAKY;
        case ELU:        return ACT_ELU;
        case GELU:       return ACT_GELU;
        case SILU:       return ACT_SILU;
        default:         return ACT_IDENTITY;
    }
}

static inline cudaStream_t pick_stream(cudaStream_t user_stream) {
    return user_stream;
}


// ---- GEMM 래퍼 (row-major 매핑) --------------------------------------------

// 단일 GEMM (row-major) TF32
static inline void gemm_rm_tf32(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha=1.f, float beta=0.f)
{
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(
        cublasGemmEx(
            h,
            /*opB,opA*/ opB, opA,
            /*m,n,k*/   N,   M,   K,
            &alpha,
            /*B*/ B, CUDA_R_32F, ldb,
            /*A*/ A, CUDA_R_32F, lda,
            &beta,
            /*C*/ C, CUDA_R_32F, ldc,
            /*computeType*/ CUBLAS_COMPUTE_32F_FAST_TF32,
            /*algo*/ CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
}

// StridedBatched GEMM (row-major) TF32
static inline void gemm_rm_strided_batched_tf32(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float* C, int ldc, long long strideC,
    int batch,
    float alpha=1.f, float beta=0.f)
{
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(
            h,
            /*opB,opA*/ opB, opA,
            /*m,n,k*/   N,   M,   K,
            &alpha,
            /*B*/ B, CUDA_R_32F, ldb, strideB,
            /*A*/ A, CUDA_R_32F, lda, strideA,
            &beta,
            /*C*/ C, CUDA_R_32F, ldc, strideC,
            /*batch*/ batch,
            /*computeType*/ CUBLAS_COMPUTE_32F_FAST_TF32,
            /*algo*/ CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
}
// -----------------------------------------------------------------------------

// 디버그용
static inline void checkCudaLast(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s: %s\n", where, cudaGetErrorString(e));
    }
}
static inline void checkCudaSync(const char* where) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][SYNC] %s: %s\n", where, cudaGetErrorString(e));
    }
}

// ones 벡터 채우기
__global__ void fill_kernel(float* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}


void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,  // ← 반드시 Activation output의 ID
    int batch_size)
{
    ensure_cublas();

    std::string grad_start_id = final_output_id;

    // 1) LOSS backward: dL/dy_pred 생성
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();
        const std::string loss_type = loss_op.extra_params.loss_type;
        const std::string label_id  = loss_op.extra_params.label_id;

        const float* y_true = tensors[label_id];
        const float* y_pred = tensors[loss_op.input_id];

        Shape shp = shapes[loss_op.input_id];  // per-sample
        const int per_sample = shp.rows * shp.cols;
        const int sz = batch_size * per_sample;

        float* dL_dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dL_dy, (size_t)sz * sizeof(float)));

        if (loss_type == "bce") {
            bce_loss_backward<<<(sz + 255)/256, 256>>>(y_true, y_pred, dL_dy, sz, batch_size);
            checkCudaLast("bce_loss_backward");
            checkCudaSync("bce_loss_backward");
        } else if (loss_type == "mse") {
            mse_loss_backward<<<(sz + 255)/256, 256>>>(y_true, y_pred, dL_dy, sz);
            checkCudaLast("mse_loss_backward");
            checkCudaSync("mse_loss_backward");
        } else {
            std::fprintf(stderr, "[LOSS][BW] unsupported: %s\n", loss_type.c_str());
        }

        grad_start_id = loss_op.input_id;
        gradients[loss_op.input_id] = dL_dy;
    }

    // 2) 나머지 역전파
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        if (op.op_type == LOSS) continue;

        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id))
                         ? tensors[op.param_id] : nullptr;
        float* grad_out_full = gradients[op.output_id];
        if (!grad_out_full && op.op_type != FLATTEN) continue;

        Shape in_shape  = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        if (out_shape.rows == 0 || out_shape.cols == 0) out_shape = in_shape;

        const int M = out_shape.rows;   // == in_shape.rows
        const int N = out_shape.cols;
        const int K = in_shape.cols;
        const int in_size  = in_shape.rows * in_shape.cols;
        const int out_size = out_shape.rows * out_shape.cols;

        float* grad_input_full = nullptr;
        if (op.op_type != FLATTEN) {
            CUDA_CHECK(cudaMalloc(&grad_input_full, (size_t)batch_size * in_size * sizeof(float)));
        }

        switch (op.op_type) {
        case MATMUL: {
            if (!param) break; // W 없음

            // dX = dY · W^T  (B, M, K)
            gemm_rm_strided_batched_tf32(
                g_cublas,
                /*transA=*/false, /*transB=*/true,
                /*M=*/M, /*N=*/K, /*K=*/N,
                /*A =*/ grad_out_full,   /*lda =*/ N, /*strideA =*/ (long long)M * N,
                /*B =*/ param,           /*ldb =*/ N, /*strideB =*/ 0LL,
                /*C =*/ grad_input_full, /*ldc =*/ K, /*strideC =*/ (long long)M * K,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // dW = sum_b (X_b^T · dY_b)
            // 1) dW_tmp[b] = X_b^T(K,M) · dY_b(M,N)  →  (B, K, N)
            float* dW_tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&dW_tmp, (size_t)batch_size * K * N * sizeof(float)));

            gemm_rm_strided_batched_tf32(
                g_cublas,
                /*transA=*/true, /*transB=*/false,
                /*M=*/K, /*N=*/N, /*K=*/M,
                /*A =*/ input,          /*lda =*/ K, /*strideA =*/ (long long)M * K,
                /*B =*/ grad_out_full,  /*ldb =*/ N, /*strideB =*/ (long long)M * N,
                /*C =*/ dW_tmp,         /*ldc =*/ N, /*strideC =*/ (long long)K * N,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // 2) 배치축 합산: ones(1,B) · dW_tmp(B, K*N) → dW(1, K*N)
            float* dW_accum = nullptr;
            CUDA_CHECK(cudaMalloc(&dW_accum, (size_t)K * N * sizeof(float)));

            float* onesB = nullptr;
            CUDA_CHECK(cudaMalloc(&onesB, (size_t)batch_size * sizeof(float)));
            {
                int thr = 256, blk = (batch_size + thr - 1) / thr;
                fill_kernel<<<blk, thr>>>(onesB, 1.0f, batch_size);
            }

            // C(1, K*N) = A(1, B) · B(B, K*N)
            gemm_rm_tf32(
                g_cublas, false, false,
                /*M=*/1, /*N=*/(K * N), /*K=*/batch_size,
                /*A=*/onesB,     /*lda=*/batch_size,
                /*B=*/dW_tmp,    /*ldb=*/(K * N),
                /*C=*/dW_accum,  /*ldc=*/(K * N),
                1.f, 0.f
            );

            gradients[op.param_id] = dW_accum;

            CUDA_CHECK(cudaFree(dW_tmp));
            CUDA_CHECK(cudaFree(onesB));
            break;
        }

        case ADD: {
            // dX = dY (그대로 복사)
            const size_t bytes = (size_t)batch_size * out_size * sizeof(float);
            CUDA_CHECK(cudaMemcpy(grad_input_full, grad_out_full, bytes, cudaMemcpyDeviceToDevice));

            // dB = sum over batch and rows → ones(1, B*M) · dY(B*M, N)
            const int rowsB = batch_size * M;
            const int cols  = N;

            float* grad_bias = nullptr;      // [cols]
            CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)cols * sizeof(float)));

            float* onesR = nullptr;
            CUDA_CHECK(cudaMalloc(&onesR, (size_t)rowsB * sizeof(float)));
            {
                int thr = 256, blk = (rowsB + thr - 1) / thr;
                fill_kernel<<<blk, thr>>>(onesR, 1.0f, rowsB);
            }

            gemm_rm_tf32(
                g_cublas, false, false,
                /*M=*/1, /*N=*/cols, /*K=*/rowsB,
                /*A=*/onesR,            /*lda=*/rowsB,
                /*B=*/grad_out_full,    /*ldb=*/cols,
                /*C=*/grad_bias,        /*ldc=*/cols,
                1.f, 0.f
            );

            gradients[op.param_id] = grad_bias;
            CUDA_CHECK(cudaFree(onesR));
            break;
        }

        // -------- 활성화 계열: launch_activation_backward 호출 --------
        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU:
        {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* gout = grad_out_full;             // dL/dout
            const float* out  = tensors[op.output_id];     // f(z)
            const float* in   = tensors[op.input_id];      // z (pre-activation)
            float* gin        = grad_input_full;           // dL/din

            const int act = map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            cudaStream_t stream = 0; // 기본 스트림

            launch_activation_backward(
                /*grad_out*/ gout,
                /*in      */ in,
                /*out     */ out,
                /*grad_in */ gin,
                /*rows    */ rowsB,
                /*cols    */ cols,
                /*act     */ act,
                /*alpha   */ alpha,
                /*gelu    */ gelu_tanh_flag,
                /*stream  */ stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        // -------- Softmax: launch_softmax_backward 호출 --------
        case SOFTMAX:
        {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* gout = grad_out_full;           // dL/dY
            const float* y    = tensors[op.output_id];   // Y = softmax(X)
            float* gin        = grad_input_full;         // dL/dX

            float temperature = (op.extra_params.temperature > 0.f)
                              ? op.extra_params.temperature : 1.f;
            cudaStream_t stream = 0;

            launch_softmax_backward(
                /*grad_out*/ gout,
                /*out     */ y,
                /*grad_in */ gin,
                /*rows    */ rowsB,
                /*cols    */ cols,
                /*temperature*/ temperature,
                /*stream  */ stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FLATTEN: {
            // shape만 바뀌는 op → 그래디언트 패스-스루
            gradients[op.input_id] = grad_out_full;
            continue;
        }

        default:
            // 다른 OpType들(MATMUL/ADD/CONV2D 등) 외 구현은 위에서 처리/유지
            break;
        }

        if (op.op_type != FLATTEN) {
            if (!grad_input_full) {
                std::fprintf(stderr, "[BW] grad_input_full is null: op=%d\n", op.op_type);
            } else {
                gradients[op.input_id] = grad_input_full;
            }
        }
    }
}