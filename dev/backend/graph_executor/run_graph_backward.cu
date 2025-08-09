// run_graph_backward.cu
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "backward_kernels_optimized.cuh"
#include "activation_backward.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss_kernels.cuh"
#include "transpose.cuh"

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

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,  // ← 반드시 Activation output의 ID여야 함
    int batch_size)
{
    std::string grad_start_id = final_output_id;

    // 1) LOSS 처리 (그래디언트 시작점)
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
            bce_loss_backward<<<(sz + 255)/256, 256>>>(
                y_true, y_pred, dL_dy, sz, batch_size);
            checkCudaLast("bce_loss_backward");
            checkCudaSync("bce_loss_backward");
        } else if (loss_type == "mse") {
            mse_loss_backward<<<(sz + 255)/256, 256>>>(
                y_true, y_pred, dL_dy, sz);
            checkCudaLast("mse_loss_backward");
            checkCudaSync("mse_loss_backward");
        } else {
            std::fprintf(stderr, "[LOSS][BW] unsupported: %s\n", loss_type.c_str());
        }

        grad_start_id = loss_op.input_id;
        gradients[loss_op.input_id] = dL_dy;
    }

    // 2) 나머지 역전파 (배치 루프 및 누적)
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

        const int in_rows  = in_shape.rows,  in_cols  = in_shape.cols;
        const int out_rows = out_shape.rows, out_cols = out_shape.cols;
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

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, (size_t)in_cols * out_cols * sizeof(float));
                cudaMemset(grad_weight, 0, (size_t)in_cols * out_cols * sizeof(float));

                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b   = grad_out_full   + b * out_stride;
                    float* grad_input_b = grad_input_full + b * in_stride;
                    float* input_b      = input           + b * in_stride;

                    float* W_T = nullptr;
                    cudaMalloc(&W_T, sizeof(float) * in_cols * out_cols);
                    launch_transpose(param, W_T, in_cols, out_cols);
                    cudaMemset(grad_input_b, 0, sizeof(float) * in_size);

                    int total_threads = out_rows * in_cols;
                    if (total_threads <= 1024) {
                        int blockSize = std::min(32, total_threads);
                        int gridSize  = (total_threads + blockSize - 1) / blockSize;
                        matmul_backward_input_simple<<<gridSize, blockSize>>>(
                            grad_out_b, W_T, grad_input_b, out_rows, out_cols, in_cols);
                        checkCudaLast("matmul_bw_input_simple");
                    } else {
                        dim3 blockDim(16, 16);
                        dim3 gridDim((in_cols + 15) / 16, (out_rows + 15) / 16);
                        matmul_backward_input_shared<<<gridDim, blockDim>>>(
                            grad_out_b, W_T, grad_input_b, out_rows, out_cols, in_cols);
                        checkCudaLast("matmul_bw_input_shared");
                    }
                    cudaFree(W_T);

                    float* input_T = nullptr;
                    cudaMalloc(&input_T, sizeof(float) * in_rows * in_cols);
                    launch_transpose(input_b, input_T, in_rows, in_cols);

                    float* grad_weight_b = nullptr;
                    cudaMalloc(&grad_weight_b, (size_t)in_cols * out_cols * sizeof(float));
                    dim3 blockDimW(16, 16);
                    dim3 gridDimW((out_cols + 15) / 16, (in_cols + 15) / 16);
                    matmul_backward_weight_shared<<<gridDimW, blockDimW>>>(
                        input_T, grad_out_b, grad_weight_b, in_cols, out_cols, in_rows);
                    checkCudaLast("matmul_bw_weight_shared");
                    cudaFree(input_T);

                    int sz = in_cols * out_cols;
                    int thr = 256;
                    int blk = (sz + thr - 1) / thr;
                    add_inplace<<<blk, thr>>>(grad_weight, grad_weight_b, sz);
                    checkCudaLast("add_inplace grad_weight");
                    cudaFree(grad_weight_b);
                }
                gradients[op.param_id] = grad_weight;
                break;
            }

            case ADD: {
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, (size_t)out_cols * sizeof(float));
                cudaMemset(grad_bias, 0, (size_t)out_cols * sizeof(float));

                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b   = grad_out_full   + b * out_stride;
                    float* grad_input_b = grad_input_full + b * in_stride;

                    add_backward_input<<<(out_size + 255)/256, 256>>>(grad_out_b, grad_input_b, out_size);
                    checkCudaLast("add_backward_input");

                    float* grad_bias_b = nullptr;
                    cudaMalloc(&grad_bias_b, (size_t)out_cols * sizeof(float));
                    add_backward_bias<<<(out_cols + 255)/256, 256>>>(grad_out_b, grad_bias_b, out_rows, out_cols);
                    checkCudaLast("add_backward_bias");

                    int thr = 256, blk = (out_cols + thr - 1) / thr;
                    add_inplace<<<blk, thr>>>(grad_bias, grad_bias_b, out_cols);
                    cudaFree(grad_bias_b);
                }
                gradients[op.param_id] = grad_bias;
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH: {
                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b   = grad_out_full   + b * out_stride;
                    float* grad_input_b = grad_input_full + b * in_stride;
                    const float* out_b  = tensors[op.output_id] + b * out_stride; // activation output

                    activation_backward<<<(out_size + 255)/256, 256>>>(
                        grad_out_b, out_b, grad_input_b, out_rows, out_cols, op.op_type);
                    checkCudaLast("activation_backward");
                }
                break;
            }

            case FLATTEN: {
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
}
