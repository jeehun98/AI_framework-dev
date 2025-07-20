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
    int total_size = shapes[final_output_id].rows * shapes[final_output_id].cols;
    float* grad_output = nullptr;
    cudaMalloc(&grad_output, total_size * sizeof(float));

    if (grad_output == nullptr) {
        std::cerr << "[ERROR] grad_output is nullptr for final_output_id = " << final_output_id << std::endl;
        return;
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    fill_gradient<<<blocks, threads>>>(grad_output, total_size, 1.0f);
    cudaDeviceSynchronize();

    if (gradients.count(final_output_id) == 0 || gradients[final_output_id] == nullptr) {
        gradients[final_output_id] = grad_output;
        std::cout << "[INFO] Manually inserted grad_output to gradients[" << final_output_id << "]\n";
    }

    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        std::cout << "[BACKWARD] op_type=" << op.op_type << ", input=" << op.input_id
                  << ", param=" << op.param_id << ", output=" << op.output_id << std::endl;

        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id)) ? tensors[op.param_id] : nullptr;
        float* grad_out = gradients[op.output_id];

        if (input == nullptr || grad_out == nullptr) {
            std::cerr << "[NULL PTR] input/output/grad_out is nullptr at op_type " << op.op_type << std::endl;
            continue;
        }

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
                    grad_out,
                    tensors[op.output_id],
                    grad_input,
                    out_rows,
                    out_cols,
                    op.op_type
                );
                break;

            case CONV2D: {
                int B = op.extra_params.batch_size;
                int H = op.extra_params.input_h;
                int W = op.extra_params.input_w;
                int KH = op.extra_params.kernel_h;
                int KW = op.extra_params.kernel_w;
                int OH = out_rows;
                int OW = out_cols;

                float* d_input = grad_input;
                float* d_kernel = nullptr;
                cudaMalloc(&d_kernel, KH * KW * sizeof(float));

                dim3 blockDim(16, 16);
                dim3 gridDimInput((W + 15) / 16, (H + 15) / 16, B);
                conv2d_backward_input_kernel<<<gridDimInput, blockDim>>>(
                    grad_out, param, d_input, B, H, W, KH, KW, OH, OW
                );

                dim3 gridDimWeight((KW + 15) / 16, (KH + 15) / 16);
                conv2d_backward_kernel_kernel<<<gridDimWeight, blockDim>>>(
                    input, grad_out, d_kernel, B, H, W, KH, KW, OH, OW
                );

                gradients[op.param_id] = d_kernel;
                break;
            }

            default:
                std::cerr << "[ERROR] Unsupported op_type: " << op.op_type << std::endl;
                break;
        }

        cudaDeviceSynchronize();
        gradients[op.input_id] = grad_input;
    }

    std::cout << "✅ run_graph_backward finished.\n";
}
