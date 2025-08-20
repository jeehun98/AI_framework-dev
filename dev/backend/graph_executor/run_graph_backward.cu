// run_graph_backward.cu (TF32 + strided-batched + fused softmax-xent, with batch-mean grads)

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "run_graph.cuh"
#include "activation_ops.cuh"
#include "softmax_kernels.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss_kernels.cuh"

#include "ge/cuda_check.cuh"
#include "ge/cublas_utils.cuh"
#include "ge/gemm_rm.cuh"
#include "ge/act_map.cuh"
#include "ge/fill.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

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

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)
{
    auto h = ge_cublas();

    std::string grad_start_id = final_output_id;
    bool fused_softmax = false;
    std::string fused_softmax_in_id, fused_softmax_out_id;

    // 1) LOSS backward: dL/dy_pred (혹은 fused면 dL/dz) 생성
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();
        const std::string loss_type = loss_op.extra_params.loss_type;
        const std::string label_id  = loss_op.extra_params.label_id;

        const float* y_true = tensors[label_id];
        const float* y_pred = tensors[loss_op.input_id];

        Shape shp = shapes[loss_op.input_id];
        const int C = shp.cols;
        const int rows_per_sample = shp.rows;          // 보통 1
        const int B = batch_size * rows_per_sample;
        const int N = B * C;

        cudaStream_t stream = 0;

        // 직전 op이 SOFTMAX인지 확인 (fused 조건)
        const OpStruct* prev = nullptr;
        if (E.size() >= 2) {
            const OpStruct& cand = E[E.size()-2];
            if (cand.op_type == SOFTMAX && cand.output_id == loss_op.input_id) {
                prev = &cand;
            }
        }

        if (loss_type == "cce" && prev) {
            // ∂L/∂z = (p - y) / B
            float* dL_dz = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dz, (size_t)N * sizeof(float)));
            launch_softmax_xent_fused_backward(y_pred, y_true, dL_dz, B, C, stream);
            checkCudaLast("launch_softmax_xent_fused_backward");
            checkCudaSync("softmax_xent_fused_backward sync");

            fused_softmax = true;
            fused_softmax_in_id  = prev->input_id;   // z
            fused_softmax_out_id = prev->output_id;  // p
            grad_start_id = prev->input_id;
            gradients[prev->input_id] = dL_dz;
        } else {
            float* dL_dy = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dy, (size_t)N * sizeof(float)));

            if (loss_type == "bce") {
                launch_bce_loss_backward(y_true, y_pred, dL_dy, N, B, stream);
                checkCudaLast("launch_bce_loss_backward");
                checkCudaSync("bce_backward sync");
            } else if (loss_type == "mse") {
                launch_mse_loss_backward(y_true, y_pred, dL_dy, N, stream);
                checkCudaLast("launch_mse_loss_backward");
                checkCudaSync("mse_backward sync");
            } else if (loss_type == "cce") {
                launch_cce_loss_backward(y_true, y_pred, dL_dy, B, C, stream);
                checkCudaLast("launch_cce_loss_backward");
                checkCudaSync("cce_backward sync");
            } else {
                std::fprintf(stderr, "[LOSS][BW] unsupported: %s\n", loss_type.c_str());
            }

            grad_start_id = loss_op.input_id;
            gradients[loss_op.input_id] = dL_dy;
        }
    }

    // 2) 나머지 역전파
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        if (op.op_type == LOSS) continue;

        if (fused_softmax && op.op_type == SOFTMAX && op.output_id == fused_softmax_out_id) {
            continue;
        }

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
            if (!param) break;

            // dX = dY · W^T  (B, M, K)
            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/false, /*transB=*/true,
                /*M=*/M, /*N=*/K, /*K=*/N,
                /*A =*/ grad_out_full,   /*lda =*/ N, /*strideA =*/ (long long)M * N,
                /*B =*/ param,           /*ldb =*/ N, /*strideB =*/ 0LL,
                /*C =*/ grad_input_full, /*ldc =*/ K, /*strideC =*/ (long long)M * K,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // dW = sum_b (X_b^T · dY_b)
            float* dW_tmp = nullptr; // (B, K, N)
            CUDA_CHECK(cudaMalloc(&dW_tmp, (size_t)batch_size * K * N * sizeof(float)));

            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/true, /*transB=*/false,
                /*M=*/K, /*N=*/N, /*K=*/M,
                /*A =*/ input,          /*lda =*/ K, /*strideA =*/ (long long)M * K,
                /*B =*/ grad_out_full,  /*ldb =*/ N, /*strideB =*/ (long long)M * N,
                /*C =*/ dW_tmp,         /*ldc =*/ N, /*strideC =*/ (long long)K * N,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // 배치축 합산
            float* dW_accum = nullptr;                 // (K, N)
            CUDA_CHECK(cudaMalloc(&dW_accum, (size_t)K * N * sizeof(float)));

            float* onesB = nullptr;
            CUDA_CHECK(cudaMalloc(&onesB, (size_t)batch_size * sizeof(float)));
            {
                int thr = 256, blk = (batch_size + thr - 1) / thr;
                ge_fill_kernel<<<blk, thr>>>(onesB, 1.0f, batch_size);
            }

            // C(1, K*N) = A(1, B) · B(B, K*N)
            gemm_rm_tf32(
                h, false, false,
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
            // dX = dY
            const size_t bytes = (size_t)batch_size * out_size * sizeof(float);
            CUDA_CHECK(cudaMemcpy(grad_input_full, grad_out_full, bytes, cudaMemcpyDeviceToDevice));

            // dB = sum over batch and rows
            const int rowsB = batch_size * M;
            const int cols  = N;

            float* grad_bias = nullptr;      // [cols]
            CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)cols * sizeof(float)));

            float* onesR = nullptr;
            CUDA_CHECK(cudaMalloc(&onesR, (size_t)rowsB * sizeof(float)));
            {
                int thr = 256, blk = (rowsB + thr - 1) / thr;
                ge_fill_kernel<<<blk, thr>>>(onesR, 1.0f, rowsB);
            }

            gemm_rm_tf32(
                h, false, false,
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

        // -------- 활성화 계열 --------
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

            const float* gout = grad_out_full;           // dL/dout
            const float* out  = tensors[op.output_id];   // f(z)
            const float* in   = tensors[op.input_id];    // z
            float* gin        = grad_input_full;         // dL/din

            const int act = ge_map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            cudaStream_t stream = 0;

            launch_activation_backward(
                gout, in, out, gin,
                rowsB, cols, act, alpha, gelu_tanh_flag, stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

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
                gout, y, gin, rowsB, cols, temperature, stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FLATTEN: {
            gradients[op.input_id] = grad_out_full;
            continue; // grad_input_full 할당 안 함
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
}
