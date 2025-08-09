// run_graph_backward.cu
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

#include "logging_config.h"               // ★ 로깅 매크로
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

static inline void checkCuda(const char* what) {
#if VERBOSE >= 2
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s: %s\n", what, cudaGetErrorString(err));
    }
#endif
}

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)
{
    std::string grad_start_id = final_output_id;

    // 1) LOSS 처리: dL/da (Option A: Sigmoid 유지)
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();

        const std::string& loss_type = loss_op.extra_params.loss_type;
        const std::string& label_id  = loss_op.extra_params.label_id;

        float* y_true = tensors[label_id];
        float* y_pred = tensors[loss_op.input_id];   // Sigmoid 출력 a

        Shape shape = shapes[loss_op.input_id];
        int sz = shape.rows * shape.cols;

        LOGV(2, "[SHAPE][LOSS] input_id=%s, shape=(%d,%d), sz=%d\n",
             loss_op.input_id.c_str(), shape.rows, shape.cols, sz);

        float* dL_dy = nullptr; // dL/da
        cudaMalloc(&dL_dy, sz * sizeof(float));

        if (loss_type == "bce") {
            // bce_loss_backward: dL/da (배치 평균 스케일 적용)
            bce_loss_backward<<<(sz + 255)/256, 256>>>(y_true, y_pred, dL_dy, sz, batch_size);
            checkCuda("bce_loss_backward launch");
#if VERBOSE >= 2
            cudaDeviceSynchronize();
#endif
        } else {
            LOGV(1, "[WARN] Unsupported loss_type: %s\n", loss_type.c_str());
        }

        // loss 입력(=activation out)부터 전파 시작
        grad_start_id = loss_op.input_id;
        gradients[loss_op.input_id]  = dL_dy;  // dL/da
        gradients[loss_op.output_id] = dL_dy;  // 동일 포인터 공유(필요 시)
    }

    // 2) 역전파 루프 (역순)
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;

        // 그래프 노드별 텐서/그라드
        float* input   = tensors[op.input_id];
        float* param   = (!op.param_id.empty() && tensors.count(op.param_id)) ? tensors[op.param_id] : nullptr;
        float* grad_out = gradients[op.output_id];        // dL/d(output)

        Shape in_shape  = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        if (out_shape.rows == 0 || out_shape.cols == 0) { // 안전 보정
            out_shape = in_shape;
        }

        int in_rows  = in_shape.rows,  in_cols  = in_shape.cols;
        int out_rows = out_shape.rows, out_cols = out_shape.cols;

        LOGV(2, "\n[SHAPE][OP] op_type=%d, output_id=%s, input_id=%s\n",
             op.op_type, op.output_id.c_str(), op.input_id.c_str());
        LOGV(2, "  input=(%d,%d) output=(%d,%d)\n",
             in_rows, in_cols, out_rows, out_cols);

        float* grad_input = nullptr;
        if (op.op_type != FLATTEN && op.op_type != LOSS) {
            cudaMalloc(&grad_input, in_rows * in_cols * sizeof(float));
        }

        if (VERBOSE >= 3 && grad_out) {
            int n = std::min(10, out_rows * out_cols);
            std::vector<float> h(n);
            cudaMemcpy(h.data(), grad_out, sizeof(float)*n, cudaMemcpyDeviceToHost);
            LOGV(3, "[DEBUG] grad_out (first %d): ", n);
            for (int i=0;i<n;++i) LOGV(3, "%.5f ", h[i]);
            LOGV(3, "\n");
        }

        switch (op.op_type) {
            case MATMUL: {
                if (!param) break;

                // dX = dY * W^T
                float* W_T = nullptr;
                cudaMalloc(&W_T, sizeof(float) * in_cols * out_cols);
                launch_transpose(param, W_T, in_cols, out_cols);
                checkCuda("transpose W->W_T");

                cudaMemset(grad_input, 0, sizeof(float) * in_rows * in_cols);

                int total = out_rows * in_cols;
                if (total <= 1024) {
                    int block = std::min(32, total);
                    int grid  = (total + block - 1) / block;
                    matmul_backward_input_simple<<<grid, block>>>(grad_out, W_T, grad_input,
                                                                  out_rows, out_cols, in_cols);
                    checkCuda("matmul_backward_input_simple");
                } else {
                    dim3 blockDim(16, 16);
                    dim3 gridDim((in_cols + 15) / 16, (out_rows + 15) / 16);
                    matmul_backward_input_shared<<<gridDim, blockDim>>>(grad_out, W_T, grad_input,
                                                                        out_rows, out_cols, in_cols);
                    checkCuda("matmul_backward_input_shared");
                }
                cudaFree(W_T);

                // dW = X^T * dY
                float* input_T = nullptr;
                cudaMalloc(&input_T, sizeof(float) * in_rows * in_cols);
                launch_transpose(input, input_T, in_rows, in_cols);
                checkCuda("transpose X->X_T");

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, in_cols * out_cols * sizeof(float));

                dim3 blockDimW(16, 16);
                dim3 gridDimW((out_cols + 15) / 16, (in_cols + 15) / 16);
                matmul_backward_weight_shared<<<gridDimW, blockDimW>>>(
                    input_T, grad_out, grad_weight, in_cols, out_cols, in_rows);
                checkCuda("matmul_backward_weight_shared");

#if VERBOSE >= 2
                cudaDeviceSynchronize();
#endif
                gradients[op.param_id] = grad_weight;
                cudaFree(input_T);
                break;
            }

            case ADD: {
                // dX = dY
                add_backward_input<<<(out_rows * out_cols + 255)/256, 256>>>(
                    grad_out, grad_input, out_rows * out_cols);
                checkCuda("add_backward_input");

                // db = reduce_rows(dY)
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, out_cols * sizeof(float));
                add_backward_bias<<<(out_cols + 255)/256, 256>>>(
                    grad_out, grad_bias, out_rows, out_cols);
                checkCuda("add_backward_bias");

#if VERBOSE >= 2
                cudaDeviceSynchronize();
#endif
                gradients[op.param_id] = grad_bias;
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH: {
                activation_backward<<<(out_rows * out_cols + 255)/256, 256>>>(
                    grad_out, tensors[op.output_id], grad_input,
                    out_rows, out_cols, op.op_type);
                checkCuda("activation_backward");
#if VERBOSE >= 2
                cudaDeviceSynchronize();
#endif

                if (VERBOSE >= 3 && grad_input) {
                    int n = std::min(10, in_rows * in_cols);
                    std::vector<float> h(n);
                    cudaMemcpy(h.data(), grad_input, sizeof(float)*n, cudaMemcpyDeviceToHost);
                    LOGV(3, "[DEBUG][POST] grad_input (first %d): ", n);
                    for (int i=0;i<n;++i) LOGV(3, "%.5f ", h[i]);
                    LOGV(3, "\n");
                }
                break;
            }

            case FLATTEN:
                // reshape 성격 → 포인터 얕은 전파
                gradients[op.input_id] = grad_out;
                break;

            case LOSS:
                // 이미 처리됨
                break;
        }

        // grad_input 저장
        if (grad_input == nullptr && op.op_type != FLATTEN && op.op_type != LOSS) {
            LOGV(1, "[ERROR] grad_input NULL for op_type=%d, input_id=%s\n",
                 op.op_type, op.input_id.c_str());
        }
        if (grad_input && op.op_type != FLATTEN && op.op_type != LOSS) {
            gradients[op.input_id] = grad_input;
        }

        // 파라미터 gradient 통계(요약)
        if (!op.param_id.empty() && gradients.count(op.param_id) && VERBOSE >= 2) {
            float* g = gradients[op.param_id];
            Shape ps = shapes[op.param_id];
            int sz = ps.rows * ps.cols;
            std::vector<float> host(sz);
            cudaMemcpy(host.data(), g, sizeof(float)*sz, cudaMemcpyDeviceToHost);

            float mn = host[0], mx = host[0], sum = 0.f;
            for (int i = 0; i < sz; ++i) {
                mn = std::min(mn, host[i]);
                mx = std::max(mx, host[i]);
                sum += host[i];
            }
            float mean = sum / std::max(sz, 1);
            std::cout << "[GRADIENT] " << op.param_id
                      << " → min=" << mn << ", max=" << mx
                      << ", mean=" << mean << std::endl;

            if (VERBOSE >= 3) {
                int k = std::min(sz, 10);
                for (int i=0;i<k;++i)
                    std::cout << "  [" << i << "] = " << host[i] << std::endl;
            }
        }
    }
}
