#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "backward_kernels_optimized.cuh"
#include "activation_backward.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"

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
    cudaError_t err = cudaMalloc(&grad_output, total_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed for grad_output: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    fill_gradient<<<blocks, threads>>>(grad_output, total_size, 1.0f);
    cudaDeviceSynchronize();

    if (gradients.count(grad_start_id) == 0 || gradients[grad_start_id] == nullptr) {
        gradients[grad_start_id] = grad_output;
    }

    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;

        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id)) ? tensors[op.param_id] : nullptr;
        float* grad_out = gradients[op.output_id];

        if (input == nullptr || grad_out == nullptr) {
            std::cerr << "[NULL PTR] input/output/grad_out is nullptr at op_type " << op.op_type << std::endl;
            continue;
        }

        float host_debug_val = 0.0f;
        cudaMemcpy(&host_debug_val, grad_out, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "[DEBUG] grad_out[0] for " << op.output_id << " = " << host_debug_val << "\n";

        Shape in_shape = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        int in_rows = in_shape.rows;
        int in_cols = in_shape.cols;
        int out_rows = out_shape.rows;
        int out_cols = out_shape.cols;

        float* grad_input = nullptr;
        cudaError_t err = cudaMalloc(&grad_input, in_rows * in_cols * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaMalloc failed for grad_input: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        switch (op.op_type) {
            case MATMUL: {
                if (param == nullptr) {
                    std::cerr << "[ERROR] param is nullptr for MATMUL backward." << std::endl;
                    cudaFree(grad_input);
                    continue;
                }

                dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
                dim3 gridDimInput((out_cols + TILE_WIDTH - 1) / TILE_WIDTH,
                                  (out_rows + TILE_WIDTH - 1) / TILE_WIDTH);
                matmul_backward_input_shared<<<gridDimInput, blockDim>>>(
                    grad_out, param, grad_input, out_rows, in_cols, out_cols);

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, in_cols * out_cols * sizeof(float));

                dim3 gridDimWeight((out_cols + TILE_WIDTH - 1) / TILE_WIDTH,
                                   (in_cols + TILE_WIDTH - 1) / TILE_WIDTH);
                matmul_backward_weight_shared<<<gridDimWeight, blockDim>>>(
                    input, grad_out, grad_weight, out_rows, out_cols, in_cols);

                float test_val = 0.0f;
                cudaMemcpy(&test_val, grad_input, sizeof(float), cudaMemcpyDeviceToHost);
                if (isnan(test_val) || isinf(test_val)) {
                    std::cerr << "[NaN DETECT] grad_input for MATMUL has NaN or Inf." << std::endl;
                }

                gradients[op.param_id] = grad_weight;
                break;
            }

            case ADD: {
                add_backward_input<<<blocks, threads>>>(grad_out, grad_input, out_rows * out_cols);
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, out_cols * sizeof(float));
                add_backward_bias<<<(out_cols + threads - 1) / threads, threads>>>(
                    grad_out, grad_bias, out_rows, out_cols);
                gradients[op.param_id] = grad_bias;
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH:
                activation_backward<<<blocks, threads>>>(
                    grad_out, tensors[op.output_id], grad_input,
                    out_rows, out_cols, op.op_type);
                break;

            case FLATTEN:
                gradients[op.input_id] = grad_out;
                cudaFree(grad_input);
                break;

            case CONV2D: {
                if (param == nullptr) {
                    std::cerr << "[ERROR] param is nullptr for CONV2D backward." << std::endl;
                    cudaFree(grad_input);
                    continue;
                }

                int B = op.extra_params.batch_size;
                int H = op.extra_params.input_h;
                int W = op.extra_params.input_w;
                int KH = op.extra_params.kernel_h;
                int KW = op.extra_params.kernel_w;
                int IC = op.extra_params.input_c;
                int OC = op.extra_params.output_c;
                int OH = out_rows;
                int OW = out_cols;

                float* d_kernel = nullptr;
                cudaMalloc(&d_kernel, OC * IC * KH * KW * sizeof(float));
                dim3 blockDim(16, 16);
                dim3 gridDimInput((W + 15) / 16, (H + 15) / 16, B);
                conv2d_backward_input_kernel<<<gridDimInput, blockDim>>>(
                    grad_out, param, grad_input, B, H, W, IC, OC, KH, KW, OH, OW);

                dim3 gridDimWeight((KW + 15) / 16, (KH + 15) / 16);
                conv2d_backward_kernel_kernel<<<gridDimWeight, blockDim>>>(
                    input, grad_out, d_kernel, B, H, W, IC, OC, KH, KW, OH, OW);

                float test_conv_val = 0.0f;
                cudaMemcpy(&test_conv_val, grad_input, sizeof(float), cudaMemcpyDeviceToHost);
                if (isnan(test_conv_val) || isinf(test_conv_val)) {
                    std::cerr << "[NaN DETECT] grad_input in CONV2D backward is NaN!" << std::endl;
                }

                gradients[op.param_id] = d_kernel;
                break;
            }

            case LOSS:
                gradients[op.input_id] = grad_out;
                cudaFree(grad_input);
                break;

            default:
                std::cerr << "[ERROR] Unsupported op_type: " << op.op_type << std::endl;
                cudaFree(grad_input);
                continue;
        }

        cudaDeviceSynchronize();
        if (op.op_type != FLATTEN && op.op_type != LOSS) {
            gradients[op.input_id] = grad_input;
        }
    }

    std::cout << "run_graph_backward finished.\n";
}
