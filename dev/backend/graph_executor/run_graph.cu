// run_graph.cu (updated: cuBLAS Strided-Batched + bias fuse + single-launch activations)
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cublas_v2.h>

#include "run_graph.cuh"
#include "activation_ops.cuh"
#include "softmax_kernels.cuh"
#include "add_bias_rowwise.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

// === cuBLAS 에러 체크 ===
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
        // Ampere+에서 TF32 쓰고 싶으면 다음 줄 주석 해제
        // cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    }
}

// ------------------- 텐서/그래디언트 헬퍼 -------------------
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

// 필요 시 출력 텐서를 새로 할당해 등록
inline float* ensure_output(std::unordered_map<std::string, uintptr_t>& tensors,
                            const std::unordered_map<std::string, Shape>& shapes,
                            const std::string& out_id,
                            const Shape& out_shape,
                            int batch_size)
{
    auto it = tensors.find(out_id);
    if (it != tensors.end()) {
        return reinterpret_cast<float*>(it->second);
    }
    size_t elems = static_cast<size_t>(batch_size) * out_shape.rows * out_shape.cols;
    float* dptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dptr, elems * sizeof(float)));
    tensors[out_id] = reinterpret_cast<uintptr_t>(dptr);
    return dptr;
}

// ------------------- 활성화 매핑 -------------------
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

// 외부에서 스트림을 관리한다면 그대로 넘기고, 없다면 0 사용 가능
static inline cudaStream_t pick_stream(cudaStream_t user_stream) {
    return user_stream; // 필요 시 nullptr/0 허용
}

// -----------------------------------------------------------------------------
// row-major + StridedBatched (A/B/C가 등간격 스트라이드로 배치 반복) - FP32 기본
static inline void gemm_rm_strided_batched(
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
        cublasSgemmStridedBatched(
            h, opB, opA, N, M, K,
            &alpha,
            B, ldb, strideB,
            A, lda, strideA,
            &beta,
            C, ldc, strideC,
            batch
        )
    );
}

// row-major + StridedBatched (TF32 fast) - 권장 경로
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

static inline float* ensure_output(std::unordered_map<std::string, float*>& tensors,
                                   std::unordered_map<std::string, Shape>& shapes,
                                   const std::string& out_id,
                                   const Shape& out_shape,
                                   int batch_size)
{
    auto it = tensors.find(out_id);
    if (it != tensors.end()) {
        shapes[out_id] = out_shape;
        return it->second;
    }
    float* out_ptr = nullptr;
    const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&out_ptr, bytes));
    tensors[out_id] = out_ptr;
    shapes[out_id] = out_shape;
    return out_ptr;
}

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size)
{
    for (size_t i = 0; i < E.size(); ++i) {
        const auto& op = E[i];
        if (op.op_type == LOSS) continue;

        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end())
                         ? tensors[op.param_id] : nullptr;

        const Shape in_shape = shapes[op.input_id];
        Shape out_shape = in_shape;

        switch (op.op_type) {
        case MATMUL: {
            if (!param) {
                fprintf(stderr, "[MATMUL] missing param for %s\n", op.output_id.c_str());
                break;
            }
            // A[M,K] * W[K,N] = C[M,N]
            const Shape w_shape = shapes[op.param_id]; // [K, N]
            const int M = in_shape.rows;
            const int K = in_shape.cols;
            const int N = w_shape.cols;
            out_shape = { M, N };

            // 다음 op가 row-wise ADD면 bias fuse
            bool fuse_bias = false;
            float* bias_ptr = nullptr;
            std::string out_id = op.output_id;

            if ((i + 1) < E.size()) {
                const auto& next = E[i + 1];
                if (next.op_type == ADD && next.input_id == op.output_id &&
                    !next.param_id.empty() && tensors.count(next.param_id))
                {
                    const Shape bshape = shapes[next.param_id];
                    const bool row_bias = (bshape.rows == 1 && bshape.cols == N) ||
                                          (bshape.rows == N && bshape.cols == 1);
                    if (row_bias) {
                        fuse_bias = true;
                        bias_ptr = tensors[next.param_id];
                        out_id = next.output_id; // ADD 출력으로 바로 기록
                    }
                }
            }

            float* Y = ensure_output(tensors, shapes, out_id, out_shape, batch_size);

            // 배치 루프 없이 GEMM 1회
            ensure_cublas();
            const long long strideA = (long long)M * K;
            const long long strideC = (long long)M * N;

            // TF32 경로
            gemm_rm_strided_batched_tf32(
                g_cublas,
                /*transA=*/false, /*transB=*/false,
                /*M=*/M, /*N=*/N, /*K=*/K,
                /*A =*/ input,              /*lda =*/ K, /*strideA =*/ strideA,
                /*B =*/ param,              /*ldb =*/ N, /*strideB =*/ 0LL, // 공유 가중치
                /*C =*/ Y,                  /*ldc =*/ N, /*strideC =*/ strideC,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // bias를 한 번에 더함 (ADD fuse)
            if (fuse_bias) {
                const int rowsB = batch_size * M;
                const int cols  = N;
                launch_add_bias_rowwise(/*input=*/Y, /*bias=*/bias_ptr, /*output=*/Y,
                                        /*rows=*/rowsB, /*cols=*/cols);
                CUDA_CHECK(cudaGetLastError());
                ++i; // 다음 ADD 스킵
            }
            break;
        }

        case ADD: {
            if (!param) {
                fprintf(stderr, "[ADD] missing param for %s\n", op.output_id.c_str());
                break;
            }
            out_shape = in_shape;
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            const Shape bshape = shapes[op.param_id];
            const bool row_bias = (bshape.rows == 1 && bshape.cols == out_shape.cols) ||
                                  (bshape.rows == out_shape.cols && bshape.cols == 1);

            if (row_bias) {
                // 배치까지 합쳐 한 번에
                const int rowsB = batch_size * out_shape.rows;
                const int cols  = out_shape.cols;
                launch_add_bias_rowwise(/*input=*/input, /*bias=*/param, /*output=*/output,
                                        /*rows=*/rowsB, /*cols=*/cols);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // 필요시 다른 add 구현 추가
                const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
                CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
                fprintf(stderr,
                        "[ADD] unsupported shape: input(%d,%d) + param(%d,%d). Expect row-wise bias.\n",
                        in_shape.rows, in_shape.cols, bshape.rows, bshape.cols);
            }
            break;
        }

        // ---------- 활성화( bias 선택 지원 ) ----------
        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU:
        {
            Shape act_shape = shapes[op.output_id];
            float* output   = ensure_output(tensors, shapes, op.output_id, act_shape, batch_size);

            const int rowsB = batch_size * act_shape.rows;
            const int cols  = act_shape.cols;

            const float* in_ptr   = input;      // pre-activation z
            const float* bias_ptr = nullptr;    // 선택적 bias
            if (!op.param_id.empty()) {
                auto it = tensors.find(op.param_id);
                if (it != tensors.end()) bias_ptr = it->second;
            }

            const int act = map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;            // Leaky/ELU
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            cudaStream_t stream = 0; // 이 함수 서명엔 stream이 없으므로 기본 스트림 사용

            launch_activation_forward(
                /*in*/   in_ptr,
                /*bias*/ bias_ptr,
                /*out*/  output,
                /*rows*/ rowsB,
                /*cols*/ cols,
                /*act*/  act,
                /*alpha*/alpha,
                /*gelu_tanh*/ gelu_tanh_flag,
                /*stream*/ stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        // ---------- Softmax(행 기준, temperature 지원) ----------
        case SOFTMAX:
        {
            Shape sm_shape = shapes[op.output_id];
            float* output  = ensure_output(tensors, shapes, op.output_id, sm_shape, batch_size);
            const float* in_ptr = input;

            const int rowsB = batch_size * sm_shape.rows;
            const int cols  = sm_shape.cols;

            float temperature = (op.extra_params.temperature > 0.f)
                                  ? op.extra_params.temperature : 1.f;
            cudaStream_t stream = 0;

            launch_softmax_forward(
                /*in*/   in_ptr,
                /*out*/  output,
                /*rows*/ rowsB,
                /*cols*/ cols,
                /*temperature*/ temperature,
                /*stream*/ stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FLATTEN: {
            // 통짜 D2D 복사
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
            CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
            break;
        }

        case CONV2D: {
            // (기존 구현 유지)
            int KH = op.extra_params.kernel_h;
            int KW = op.extra_params.kernel_w;
            int SH = op.extra_params.stride_h;
            int SW = op.extra_params.stride_w;
            int PH = op.extra_params.padding_h;
            int PW = op.extra_params.padding_w;
            int IH = op.extra_params.input_h;
            int IW = op.extra_params.input_w;
            int IC = op.extra_params.input_c;
            int OC = op.extra_params.output_c;

            const int OW = shapes[op.output_id].cols / OC;
            const int OH = shapes[op.output_id].rows;

            float* output = ensure_output(tensors, shapes, op.output_id, shapes[op.output_id], batch_size);

            dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
            dim3 gridDim((OW + TILE_WIDTH - 1) / TILE_WIDTH,
                         (OH + TILE_WIDTH - 1) / TILE_WIDTH,
                         OC);

            for (int b = 0; b < batch_size; ++b) {
                float* in_b  = input  + (size_t)b * IH * IW * IC;
                float* out_b = output + (size_t)b * OH * OW * OC;
                conv2d_forward_kernel<<<gridDim, blockDim>>>(
                    in_b, param, out_b,
                    /*batch_size=*/1, IH, IW,
                    IC, OC,
                    KH, KW,
                    OH, OW
                );
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        default:
            fprintf(stderr, "[ERROR] Unsupported op_type: %d\n", op.op_type);
            break;
        } // switch
    } // for i

    // 최종 출력 호스트로 복사
    const Shape out_shape = shapes[final_output_id];
    const size_t out_bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
    
    // 변경
    if (out_host != nullptr) {
        CUDA_CHECK(cudaMemcpy(out_host, tensors[final_output_id], out_bytes, cudaMemcpyDeviceToHost));
    }
}