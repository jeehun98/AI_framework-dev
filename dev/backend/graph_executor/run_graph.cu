#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cublas_v2.h>    

#include "run_graph.cuh"
#include "matmul_tiled.cuh"          // ✅ 새 GEMM
#include "activation_ops.cuh"
#include "add_bias_rowwise.cuh"      // ✅ 새 row-wise bias add
#include "cnn_kernels.cuh"
#include "op_structs.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16   // conv 등 다른 커널의 블록 크기 용도. (matmul_tiled는 자체 내부에서 32 사용)
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
        // Ampere+면 TF32 활성화하고 싶다면:
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
    float alpha=1.f, float beta=0.f) {

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
        // shape 갱신(필요 시)
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
    // index 기반 루프(다음 ADD를 fuse하면 건너뛰기 위해)
    for (size_t i = 0; i < E.size(); ++i) {
        const auto& op = E[i];

        // LOSS 노드는 forward 경로에선 패스 (손실 전용 실행에서 처리)
        if (op.op_type == LOSS) continue;

        // 공통 입력/파라미터
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end())
                         ? tensors[op.param_id] : nullptr;

        const Shape in_shape = shapes[op.input_id];
        Shape out_shape = in_shape;  // 기본은 동일 (op별로 갱신)

        // 활성화/플랫/기타용 런치 설정
        const int rows_in  = in_shape.rows;
        const int cols_in  = in_shape.cols;
        const int total_in = rows_in * cols_in;
        const int act_threads = 256;
        const int act_blocks  = (total_in + act_threads - 1) / act_threads;

        switch (op.op_type) {
        case MATMUL: {
            if (!param) {
                fprintf(stderr, "[MATMUL] missing param for %s\n", op.output_id.c_str());
                break;
            }
            // A[M,K] * W[K,N] = C[M,N]
            const Shape w_shape = shapes[op.param_id]; // [K, N]
            const int M = in_shape.rows;
            const int K = in_shape.cols;     // == w_shape.rows
            const int N = w_shape.cols;
            out_shape = { M, N };

            // --- Fuse candidate: 다음 op가 ADD이고, 입력이 여기 출력이며 bias shape가 (1,N) 혹은 (N,1) ---
            bool fused = false;
            float* bias_ptr = nullptr;
            std::string fused_out_id = op.output_id; // 기본은 matmul 출력 ID

            if (i + 1 < E.size()) {
                const auto& next = E[i + 1];
                if (next.op_type == ADD && next.input_id == op.output_id &&
                    !next.param_id.empty() && tensors.count(next.param_id))
                {
                    const Shape bshape = shapes[next.param_id];
                    const bool row_bias = (bshape.rows == 1 && bshape.cols == N) ||
                                          (bshape.rows == N && bshape.cols == 1);
                    if (row_bias) {
                        fused = true;
                        bias_ptr = tensors[next.param_id];
                        fused_out_id = next.output_id;  // 최종 출력은 ADD의 output으로
                    }
                }
            }

            // 출력 버퍼 준비 (fused면 다음 ADD의 output에 바로 기록)
            float* output = ensure_output(tensors, shapes, fused ? fused_out_id : op.output_id, out_shape, batch_size);

            // 배치 루프
            const size_t in_stride  = (size_t)in_shape.rows * in_shape.cols;
            const size_t out_stride = (size_t)out_shape.rows * out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* A = input + b * in_stride;   // [M,K]
                const float* W = param;                   // [K,N]
                float* C       = output + b * out_stride; // [M,N]

                if (fused) {
                    launch_matmul_bias_tiled(A, W, bias_ptr, C, M, K, N /*, stream=0*/);
                } else {
                    launch_matmul_tiled(A, W, C, M, K, N /*, stream=0*/);
                }
            }

            // fused였다면 다음 ADD는 스킵
            if (fused) { ++i; }
            break;
        }

        case ADD: {
            // 여기 오는 ADD는 (1) fuse 못 했거나 (2) 다른 종류의 add
            if (!param) {
                fprintf(stderr, "[ADD] missing param for %s\n", op.output_id.c_str());
                break;
            }

            // 기본적으로 shape 유지
            out_shape = in_shape;
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            // bias row-wise인지 확인
            const Shape bshape = shapes[op.param_id];
            const bool row_bias = (bshape.rows == 1 && bshape.cols == out_shape.cols) ||
                                  (bshape.rows == out_shape.cols && bshape.cols == 1);

            const size_t stride = (size_t)out_shape.rows * out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* in_b  = input  + b * stride;
                float*       out_b = output + b * stride;

                if (row_bias) {
                    launch_add_bias_rowwise(in_b, param, out_b, out_shape.rows, out_shape.cols);
                } else {
                    // (필요 시) 원소별 add 등으로 확장 가능. 일단 에러/폴백 처리.
                    fprintf(stderr,
                            "[ADD] unsupported shape: input(%d,%d) + param(%d,%d). Expect row-wise bias.\n",
                            in_shape.rows, in_shape.cols, bshape.rows, bshape.cols);
                    // 안전 폴백: 그냥 복사 (실제 add는 수행 안 함)
                    CUDA_CHECK(cudaMemcpy(out_b, in_b, stride * sizeof(float), cudaMemcpyDeviceToDevice));
                }
            }
            break;
        }
        case SIGMOID: {
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t stride = (size_t)out_shape.rows * out_shape.cols;
            const int rows = out_shape.rows;
            const int cols = out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* in_b  = input  + b * stride;
                float*       out_b = output + b * stride;

                // bias가 별도 노드(ADD)로 이미 처리되므로 여기서는 nullptr
                launch_activation_forward(in_b, /*bias=*/nullptr, out_b,
                                          rows, cols, ACT_SIGMOID /*=3*/);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        case RELU: {
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t stride = (size_t)out_shape.rows * out_shape.cols;
            const int rows = out_shape.rows;
            const int cols = out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* in_b  = input  + b * stride;
                float*       out_b = output + b * stride;

                launch_activation_forward(in_b, /*bias=*/nullptr, out_b,
                                          rows, cols, ACT_RELU /*=2*/);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        case TANH: {
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t stride = (size_t)out_shape.rows * out_shape.cols;
            const int rows = out_shape.rows;
            const int cols = out_shape.cols;

            for (int b = 0; b < batch_size; ++b) {
                const float* in_b  = input  + b * stride;
                float*       out_b = output + b * stride;

                launch_activation_forward(in_b, /*bias=*/nullptr, out_b,
                                          rows, cols, ACT_TANH /*=4*/);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }


        case FLATTEN: {
            // 단순 패스(메모리 레이아웃만 유지) – 필요 시 reshape 정도만
            float* output = ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);
            const size_t bytes = (size_t)out_shape.rows * out_shape.cols * sizeof(float);
            for (int b = 0; b < batch_size; ++b) {
                float* in_b  = input  + (size_t)b * out_shape.rows * out_shape.cols;
                float* out_b = output + (size_t)b * out_shape.rows * out_shape.cols;
                CUDA_CHECK(cudaMemcpy(out_b, in_b, bytes, cudaMemcpyDeviceToDevice));
            }
            break;
        }

        case CONV2D: {
            // 기존 구현 유지 (배치 루프에서 한 샘플씩)
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

            // out_shape는 미리 shapes에 기록되어 있다고 가정
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
