#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cublas_v2.h>

#include "run_graph.cuh"
// (폴백/비교용으로 남겨두고 싶으면 유지, 아니면 제거 가능)
#include "matmul_tiled.cuh"
#include "activation_ops.cuh"
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

// === 전역 cuBLAS 핸들 재사용 ===
static cublasHandle_t g_cublas = nullptr;

static void ensure_cublas() {
    if (!g_cublas) {
        cublasCreate(&g_cublas);
        // Ampere+에서 TF32 쓰고 싶으면:
        // cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    }
}

// row-major + StridedBatched (A/B/C가 등간격 스트라이드로 배치 반복)
static inline void gemm_rm_strided_batched(cublasHandle_t h,
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
    // column-major 매핑: (opB,opA), (N,M,K)
    cublasSgemmStridedBatched(
        h, opB, opA, N, M, K,
        &alpha,
        B, ldb, strideB,
        A, lda, strideA,
        &beta,
        C, ldc, strideC,
        batch
    );
}

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

            gemm_rm_strided_batched(
                g_cublas,
                /*transA=*/false, /*transB=*/false,
                /*M=*/M, /*N=*/N, /*K=*/K,
                /*A =*/ tensors[op.input_id], /*lda =*/ K, /*strideA =*/ strideA,
                /*B =*/ param,                /*ldb =*/ N, /*strideB =*/ 0LL, // 공유 가중치
                /*C =*/ Y,                    /*ldc =*/ N, /*strideC =*/ strideC,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // bias를 한 번에 더함 (add_kernel 대신 launch_add_bias_rowwise 사용)
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

        case SIGMOID:
        case RELU:
        case TANH: {
            // (간단히) 배치 루프 — 필요하면 rowsB로 싱글 런치로 바꿀 수 있음
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t stride = (size_t)out_shape.rows * out_shape.cols;
            const int rows = out_shape.rows;
            const int cols = out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* in_b  = input  + b * stride;
                float*       out_b = output + b * stride;

                int act = (op.op_type == SIGMOID) ? ACT_SIGMOID
                        : (op.op_type == RELU)    ? ACT_RELU
                        :                            ACT_TANH;
                launch_activation_forward(in_b, /*bias=*/nullptr, out_b, rows, cols, act);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        case FLATTEN: {
            // 현재는 D2D 복사 (필요시 view/alias로 최적화 가능)
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t per = (size_t)out_shape.rows * out_shape.cols;
            const size_t bytes = per * sizeof(float);
            for (int b = 0; b < batch_size; ++b) {
                float* in_b  = input  + b * per;
                float* out_b = output + b * per;
                CUDA_CHECK(cudaMemcpy(out_b, in_b, bytes, cudaMemcpyDeviceToDevice));
            }
            break;
        }

        case CONV2D: {
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
    CUDA_CHECK(cudaMemcpy(out_host, tensors[final_output_id], out_bytes, cudaMemcpyDeviceToHost));
}
