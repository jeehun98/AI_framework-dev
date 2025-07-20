#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "matmul_shared_optimized_kernel.cuh"
#include "activation.cuh"
#include "add_kernel.cuh"
#include "cnn_kernels.cuh"  // ✅ CNN 연산 추가 포함
#include "op_structs.cuh"

#define TILE_WIDTH 16

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size)
{
    for (const auto& op : E) {
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end())
                         ? tensors[op.param_id] : nullptr;
        float* output;

        Shape in_shape = shapes[op.input_id];
        Shape out_shape;

        if (op.op_type == MATMUL && param != nullptr) {
            Shape w_shape = shapes[op.param_id];
            out_shape = {in_shape.rows, w_shape.cols};
        } else {
            out_shape = in_shape;
        }

        if (tensors.find(op.output_id) == tensors.end()) {
            float* out_ptr;
            cudaMalloc(&out_ptr, batch_size * out_shape.rows * out_shape.cols * sizeof(float));
            tensors[op.output_id] = out_ptr;
            shapes[op.output_id] = out_shape;
        }

        output = tensors[op.output_id];
        int rows = out_shape.rows;
        int cols = out_shape.cols;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                     (rows + TILE_WIDTH - 1) / TILE_WIDTH);
        int total = rows * cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        std::cout << "\n=== [CUDA EXECUTE] op_type: " << op.op_type
                  << " | input: " << op.input_id
                  << " | param: " << op.param_id
                  << " | output: " << op.output_id << " ===\n";

        for (int b = 0; b < batch_size; ++b) {
            float* input_b = input + b * in_shape.rows * in_shape.cols;
            float* output_b = output + b * out_shape.rows * out_shape.cols;
            const float* bias_or_param_b = param;

            switch (op.op_type) {
                case MATMUL:
                    matmul_shared_kernel_coalesced<<<dimGrid, dimBlock>>>(
                        input_b, bias_or_param_b, output_b,
                        rows, in_shape.cols, cols);
                    break;

                case ADD:
                    add_kernel<<<blocks, threads>>>(
                        input_b, bias_or_param_b, output_b,
                        rows, cols);
                    break;

                case SIGMOID:
                    activation_sigmoid<<<blocks, threads>>>(
                        input_b, bias_or_param_b, output_b,
                        rows, cols);
                    break;

                case RELU:
                    activation_relu<<<blocks, threads>>>(
                        input_b, bias_or_param_b, output_b,
                        rows, cols);
                    break;

                case TANH:
                    activation_tanh<<<blocks, threads>>>(
                        input_b, bias_or_param_b, output_b,
                        rows, cols);
                    break;

                case FLATTEN:
                    cudaMemcpy(output_b, input_b,
                               rows * cols * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                    break;

                case CONV2D: {
                    int KH = op.extra_params.kernel_h;
                    int KW = op.extra_params.kernel_w;
                    int SH = op.extra_params.stride_h;
                    int SW = op.extra_params.stride_w;
                    int PH = op.extra_params.padding_h;
                    int PW = op.extra_params.padding_w;
                    int IH = op.extra_params.input_h;
                    int IW = op.extra_params.input_w;
                    int OH = (IH + 2 * PH - KH) / SH + 1;
                    int OW = (IW + 2 * PW - KW) / SW + 1;

                    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
                    dim3 gridDim((OW + TILE_WIDTH - 1) / TILE_WIDTH,
                                (OH + TILE_WIDTH - 1) / TILE_WIDTH,
                                batch_size);

                    conv2d_forward_kernel<<<gridDim, blockDim>>>(
                        input, param, output,
                        batch_size, IH, IW,
                        KH, KW, OH, OW);
                    break;
                }


                default:
                    std::cerr << "[ERROR] Unsupported op_type: " << op.op_type << std::endl;
                    break;
            }

            cudaDeviceSynchronize();
        }
    }

    Shape out_shape = shapes[final_output_id];
    cudaMemcpy(out_host,
               tensors[final_output_id],
               batch_size * out_shape.rows * out_shape.cols * sizeof(float),
               cudaMemcpyDeviceToHost);
}
