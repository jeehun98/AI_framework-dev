// run_graph_backward.cu (updated)
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cublas_v2.h>                    // ✅ cuBLAS

#include "run_graph.cuh"
// ↓ 아래 3개는 더이상 필요 없음: backward matmul/transpose를 cuBLAS로 대체
// #include "backward_kernels_optimized.cuh"
// #include "transpose.cuh"
#include "activation_ops.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss_kernels.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void add_inplace(float* dst, const float* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dst[i] += src[i];
}

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

// ✅ cuBLAS 에러 헬퍼
static inline void CUBLAS_CHECK(cublasStatus_t s, const char* where) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS][ERR] %s: code=%d\n", where, int(s));
    }
}

/**
 * Row-major 행렬을 대상으로 하는 얇은 GEMM 래퍼.
 * 우리가 원하는 C(MxN)=A(MxK)*B(KxN)을 column-major인 cuBLAS로 호출하기 위해
 * 'B, A' 순서로 뒤집어 넣는다.
 */
static inline void gemm_rm(cublasHandle_t h,
                           bool transA, bool transB,
                           int M, int N, int K,
                           const float* A, int lda,   // row-major: lda=열 개수
                           const float* B, int ldb,
                           float* C, int ldc,
                           float alpha=1.0f, float beta=0.0f)
{
    // row-major에서의 전치 플래그
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // column-major로 바꿔 호출: C^T = B^T * A^T
    // => m=N, n=M, k=K, (opB,opA), (B,A), leading dims는 row-major의 열 개수
    CUBLAS_CHECK(
        cublasSgemm(h,
                    opB, opA,
                    /*m=*/N, /*n=*/M, /*k=*/K,
                    &alpha,
                    B, ldb,
                    A, lda,
                    &beta,
                    C, ldc),
        "cublasSgemm");
}

// ==== ADD backward: dX = dY, dB = sum_rows(dY) ===============================
static __global__ void add_backward_input(const float* __restrict__ grad_out,
                                          float* __restrict__ grad_in,
                                          int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) grad_in[i] = grad_out[i];  // dX = dY
}

// grad_bias[col] = sum_{row=0..rows-1} grad_out[row, col]
static __global__ void add_backward_bias(const float* __restrict__ grad_out,
                                         float* __restrict__ grad_bias,
                                         int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float s = 0.f;
    // grad_out 은 row-major [rows, cols]
    for (int r = 0; r < rows; ++r) {
        s += grad_out[r * cols + col];
    }
    grad_bias[col] = s;  // 호출 측에서 batch는 바깥 루프로 누적(add_inplace) 처리
}


void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,  // ← 반드시 Activation output의 ID여야 함
    int batch_size)
{
    // cuBLAS 핸들
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate");

    std::string grad_start_id = final_output_id;

    // 1) LOSS 처리: dL/dy_pred 생성
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();

        std::string loss_type = loss_op.extra_params.loss_type;
        std::string label_id  = loss_op.extra_params.label_id;
        const float* y_true   = tensors[label_id];
        const float* y_pred   = tensors[loss_op.input_id];  // ★ Activation 출력이어야 함

        Shape shp = shapes[loss_op.input_id];  // per-sample
        int per_sample = shp.rows * shp.cols;
        int sz = batch_size * per_sample;

        float* dL_dy = nullptr;
        cudaMalloc(&dL_dy, sz * sizeof(float));

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

        const int in_rows  = in_shape.rows,  in_cols  = in_shape.cols;   // X: [M,K]
        const int out_rows = out_shape.rows, out_cols = out_shape.cols;  // Y: [M,N]
        const int in_size  = in_rows * in_cols;
        const int out_size = out_rows * out_cols;

        const size_t in_stride  = (size_t)in_size;
        const size_t out_stride = (size_t)out_size;

        float* grad_input_full = nullptr;
        if (op.op_type != FLATTEN) {
            cudaMalloc(&grad_input_full, (size_t)batch_size * in_size * sizeof(float));
        }

        switch (op.op_type) {
        case MATMUL: {
            if (!param) break;

            // Shapes (row-major):
            // Forward:  Y[M,N] = X[M,K] * W[K,N]
            // Backward:
            //   dX[M,K] = dY[M,N] * W^T[N,K]
            //   dW[K,N] = X^T[K,M] * dY[M,N]   (배치 전체 합)
            const int M = in_rows;
            const int K = in_cols;
            const int N = out_cols;

            // dW 누적 버퍼
            float* grad_weight = nullptr;
            cudaMalloc(&grad_weight, (size_t)K * N * sizeof(float));
            cudaMemset(grad_weight, 0, (size_t)K * N * sizeof(float));

            // 배치 루프
            for (int b = 0; b < batch_size; ++b) {
                float* grad_out_b   = grad_out_full   + b * out_stride; // [M,N]
                float* grad_input_b = grad_input_full + b * in_stride;  // [M,K]
                float* input_b      = input           + b * in_stride;  // [M,K]

                // dX = dY * W^T
                // lda/ldb/ldc는 row-major의 '열 개수'
                gemm_rm(handle,
                        /*transA=*/false, /*transB=*/true,
                        /*M=*/M, /*N=*/K, /*K=*/N,
                        /*A=*/grad_out_b, /*lda=*/N,
                        /*B=*/param,      /*ldb=*/N,   // W[K,N], transB=True로 W^T 사용
                        /*C=*/grad_input_b,/*ldc=*/K,
                        /*alpha=*/1.f, /*beta=*/0.f);

                // dW += X^T * dY
                gemm_rm(handle,
                        /*transA=*/true,  /*transB=*/false,
                        /*M=*/K, /*N=*/N, /*K=*/M,
                        /*A=*/input_b,    /*lda=*/K,   // X[M,K], transA=True로 X^T
                        /*B=*/grad_out_b, /*ldb=*/N,   // dY[M,N]
                        /*C=*/grad_weight,/*ldc=*/N,
                        /*alpha=*/1.f, /*beta=*/1.f);  // 누적
            }

            gradients[op.param_id] = grad_weight;  // dW
            break;
        }

        case ADD: {
            // y = x + b(row-wise), backward:
            //   dX = dY
            //   dB = sum_rows(dY), 배치 전체 합
            float* grad_bias = nullptr;
            cudaMalloc(&grad_bias, (size_t)out_cols * sizeof(float));
            cudaMemset(grad_bias, 0, (size_t)out_cols * sizeof(float));

            // 임시 버퍼 1개만 재사용
            float* grad_bias_b = nullptr;
            cudaMalloc(&grad_bias_b, (size_t)out_cols * sizeof(float));

            for (int b = 0; b < batch_size; ++b) {
                float* grad_out_b   = grad_out_full   + b * out_stride;
                float* grad_input_b = grad_input_full + b * in_stride;

                // dX = dY
                add_backward_input<<<(out_size + 255)/256, 256>>>(grad_out_b, grad_input_b, out_size);
                checkCudaLast("add_backward_input");

                // dB_b = sum_rows(dY)
                cudaMemset(grad_bias_b, 0, (size_t)out_cols * sizeof(float));
                add_backward_bias<<<(out_cols + 255)/256, 256>>>(grad_out_b, grad_bias_b, out_rows, out_cols);
                checkCudaLast("add_backward_bias");

                // 누적
                int thr = 256, blk = (out_cols + thr - 1) / thr;
                add_inplace<<<blk, thr>>>(grad_bias, grad_bias_b, out_cols);
                checkCudaLast("add_inplace grad_bias");
            }
            cudaFree(grad_bias_b);

            gradients[op.param_id] = grad_bias;
            break;
        }

        case SIGMOID:
        case RELU:
        case TANH: {
            // rows' = batch_size * out_rows, cols' = out_cols
            const int rowsB = batch_size * out_rows;
            const int colsB = out_cols;

            // grad_out_full / grad_input_full / tensors[op.output_id] 는
            // 배치가 연속 저장이므로 그대로 전달하면 OK
            launch_activation_backward(
                /*grad_out=*/grad_out_full,
                /*out=*/tensors[op.output_id],
                /*grad_in=*/grad_input_full,
                rowsB, colsB, op.op_type);
            checkCudaLast("activation_backward");
            break;
        }


        case FLATTEN: {
            // 단순 전달
            gradients[op.input_id] = grad_out_full;
            continue;
        }

        default:
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

    cublasDestroy(handle);
}
