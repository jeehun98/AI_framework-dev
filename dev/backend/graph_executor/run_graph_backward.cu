// run_graph_backward.cu (transpose 통합 포함)

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

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)
{
    std::string grad_start_id = final_output_id;
    if (!E.empty() && E.back().op_type == LOSS) {
        grad_start_id = E.back().output_id;
    }

    int total_size = shapes[grad_start_id].rows * shapes[grad_start_id].cols;
    float* grad_output = nullptr;
    cudaMalloc(&grad_output, total_size * sizeof(float));
    fill_gradient<<<(total_size + 255)/256, 256>>>(grad_output, total_size, 1.0f);
    cudaDeviceSynchronize();

    gradients[grad_start_id] = grad_output;

    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id)) ? tensors[op.param_id] : nullptr;
        float* grad_out = gradients[op.output_id];

        Shape in_shape = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        int in_rows = in_shape.rows, in_cols = in_shape.cols;
        int out_rows = out_shape.rows, out_cols = out_shape.cols;

        float* grad_input = nullptr;
        if (op.op_type != FLATTEN && op.op_type != LOSS)
            cudaMalloc(&grad_input, in_rows * in_cols * sizeof(float));

        switch (op.op_type) {
            case MATMUL: {
                if (!param) break;

                // printf("[MATMUL_BACKWARD] out_rows=%d, out_cols=%d, in_rows=%d, in_cols=%d\n",                    out_rows, out_cols, in_rows, in_cols);

                // === 1. grad_input = grad_out @ W.T ===
                float* W_T = nullptr;
                cudaMalloc(&W_T, sizeof(float) * in_cols * out_cols);
                launch_transpose(param, W_T, in_cols, out_cols);  // [in_cols, out_cols] → [out_cols, in_cols]

                float* grad_input = nullptr;
                cudaMalloc(&grad_input, sizeof(float) * out_rows * in_cols);
                cudaMemset(grad_input, 0, sizeof(float) * out_rows * in_cols);  // atomicX 사용시 초기화 필수

                int total_threads = out_rows * in_cols;
                if (total_threads <= 1024) {
                    // ===== Simple fallback kernel =====
                    int blockSize = std::min(32, total_threads);
                    int gridSize = (total_threads + blockSize - 1) / blockSize;

                    // printf("[MATMUL_BACKWARD_INPUT_SIMPLE] total_threads=%d, blockSize=%d, gridSize=%d\n",                        total_threads, blockSize, gridSize);

                    if (blockSize > 0 && gridSize > 0) {
                        matmul_backward_input_simple<<<gridSize, dim3(blockSize)>>>(
                            grad_out, W_T, grad_input, out_rows, out_cols, in_cols);
                    } else {
                        // printf("[ERROR] Invalid launch config for simple kernel: grid=%d, block=%d\n",                            gridSize, blockSize);
                    }
                } else {
                    // ===== Shared memory kernel =====
                    dim3 blockDim(16, 16);
                    dim3 gridDim((in_cols + 15) / 16, (out_rows + 15) / 16);
                    matmul_backward_input_shared<<<gridDim, blockDim>>>(grad_out, W_T, grad_input, out_rows, out_cols, in_cols);
                    //printf("[MATMUL_BACKWARD_INPUT_SHARED] grid=(%d,%d), block=(%d,%d)\n",                        gridDim.x, gridDim.y, blockDim.x, blockDim.y);
                }

                cudaError_t err_input = cudaGetLastError();
                if (err_input != cudaSuccess) {
                    // printf("[MATMUL_BACKWARD_INPUT] Kernel launch failed: %s\n", cudaGetErrorString(err_input));
                }

                cudaFree(W_T);  // W_T 사용 완료

                // === 2. grad_weight = input.T @ grad_out ===
                float* input_T = nullptr;
                cudaMalloc(&input_T, sizeof(float) * in_rows * in_cols);
                launch_transpose(input, input_T, in_rows, in_cols);  // [in_rows, in_cols] → [in_cols, in_rows]

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, in_cols * out_cols * sizeof(float));

                dim3 blockDimW(16, 16);
                dim3 gridDimW((out_cols + 15) / 16, (in_cols + 15) / 16);

                //printf("[MATMUL_BACKWARD_WEIGHT] launching with M=%d, N=%d, K=%d\n", in_rows, out_cols, in_cols);
                // printf("[MATMUL_BACKWARD_WEIGHT] grid=(%d,%d), block=(%d,%d)\n", gridDimW.x, gridDimW.y, blockDimW.x, blockDimW.y);

                matmul_backward_weight_shared<<<gridDimW, blockDimW>>>(
                    input_T, grad_out, grad_weight, in_cols, out_cols, in_rows);  // [K x M] @ [M x N] = [K x N]

                cudaDeviceSynchronize();  // 디버깅 목적

                cudaError_t err_weight = cudaGetLastError();
                if (err_weight != cudaSuccess) {
                    printf("[MATMUL_BACKWARD_WEIGHT] Kernel launch failed: %s\n", cudaGetErrorString(err_weight));
                }

                gradients[op.param_id] = grad_weight;
                cudaFree(input_T);
                break;
            }


            case ADD: {
                add_backward_input<<<(out_rows*out_cols + 255)/256, 256>>>(grad_out, grad_input, out_rows*out_cols);
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, out_cols * sizeof(float));
                add_backward_bias<<<(out_cols + 255)/256, 256>>>(grad_out, grad_bias, out_rows, out_cols);
                gradients[op.param_id] = grad_bias;
                break;
            }


            case SIGMOID:
            case RELU:
            case TANH:
                activation_backward<<<(out_rows*out_cols + 255)/256, 256>>>(grad_out, tensors[op.output_id], grad_input, out_rows, out_cols, op.op_type);
                break;
            case FLATTEN:
                gradients[op.input_id] = grad_out;
                break;
            case CONV2D: {
                if (!param) break;
                int B = batch_size;
                int H = op.extra_params.input_h;
                int W = op.extra_params.input_w;
                int KH = op.extra_params.kernel_h;
                int KW = op.extra_params.kernel_w;
                int IC = op.extra_params.input_c;
                int OC = op.extra_params.output_c;
                int OH = out_rows;
                int OW = out_cols / OC;

                float* d_kernel = nullptr;
                cudaMalloc(&d_kernel, OC * IC * KH * KW * sizeof(float));
                conv2d_backward_input_kernel<<<dim3((W+15)/16, (H+15)/16, B), dim3(16,16)>>>(
                    grad_out, param, grad_input, B, H, W, IC, OC, KH, KW, OH, OW);
                conv2d_backward_kernel_kernel<<<dim3((KW+15)/16, (KH+15)/16), dim3(16,16)>>>(
                    input, grad_out, d_kernel, B, H, W, IC, OC, KH, KW, OH, OW);
                gradients[op.param_id] = d_kernel;
                break;
            }
            case LOSS: {
                std::string loss_type = op.extra_params.loss_type;
                std::string label_id = op.extra_params.label_id;
                float* y_true = tensors[label_id];
                float* y_pred = tensors[op.input_id];

                Shape shape = shapes[op.input_id];
                int sz = shape.rows * shape.cols;
                float* dL_dy = nullptr;
                cudaMalloc(&dL_dy, sz * sizeof(float));
                if (loss_type == "bce") {
                    bce_loss_backward<<<(sz+255)/256, 256>>>(y_true, y_pred, dL_dy, sz);

                    // 💡 Debug: print some loss gradient stats
                    float* host_loss_grad = new float[sz];
                    cudaMemcpy(host_loss_grad, dL_dy, sizeof(float) * sz, cudaMemcpyDeviceToHost);
                    float loss_sum = 0.0f;
                    for (int i = 0; i < sz; ++i) loss_sum += fabsf(host_loss_grad[i]);
                    float avg_loss_grad = loss_sum / sz;
                    std::cout << "[LOSS_GRAD] Average dL/dy: " << avg_loss_grad << std::endl;
                    delete[] host_loss_grad;
                }
                gradients[op.input_id] = dL_dy;
                gradients[op.output_id] = dL_dy;
                break;
            }
        }

        cudaDeviceSynchronize();
        if (op.op_type != FLATTEN && op.op_type != LOSS)
            gradients[op.input_id] = grad_input;

        if (!op.param_id.empty() && gradients.count(op.param_id)) {
            float* grad = gradients[op.param_id];
            Shape shape = shapes[op.param_id];
            int size = shape.rows * shape.cols;
            float* host_grad = new float[size];
            cudaMemcpy(host_grad, grad, sizeof(float) * size, cudaMemcpyDeviceToHost);
            float min_val = host_grad[0], max_val = host_grad[0], sum = 0.0f;
            for (int i = 0; i < size; ++i) {
                min_val = fminf(min_val, host_grad[i]);
                max_val = fmaxf(max_val, host_grad[i]);
                sum += host_grad[i];
            }
            float mean = sum / size;
            // 가중치 기울기 통계 출력
            std::cout << "[GRADIENT] " << op.param_id << " grad \u2192 min=" << min_val << ", max=" << max_val << ", mean=" << mean << std::endl;
            delete[] host_grad;
        }
    }
}
