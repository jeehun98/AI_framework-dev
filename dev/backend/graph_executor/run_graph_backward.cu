// run_graph_backward.cu
#include <iostream>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "run_graph.cuh"
#include "backward_kernels.cuh"
#include "activation_backward.cuh"
__global__ void transpose(float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
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
    // ✅ 최종 출력의 gradient 수동 삽입 (없을 경우)
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

    // 역순으로 연산 그래프 순회 (backward)
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
                matmul_backward_input<<<blocks, threads>>>(
                    grad_out, param, grad_input, out_rows, in_cols, out_cols);

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, in_cols * out_cols * sizeof(float));

                float* input_T = nullptr;
                cudaMalloc(&input_T, in_cols * in_rows * sizeof(float));

                // ✅ transpose input → input_T
                dim3 blockDimT(16, 16);
                dim3 gridDimT((in_cols + 15) / 16, (in_rows + 15) / 16);
                transpose<<<gridDimT, blockDimT>>>(input, input_T, in_rows, in_cols);

                // ✅ weight gradient 계산
                matmul_backward_weight<<<blocks, threads>>>(
                    input_T, grad_out, grad_weight, in_rows, in_cols, out_cols);

                gradients[op.param_id] = grad_weight;

                cudaFree(input_T);
                break;
            }


            case ADD: {
                // grad_input = grad_out
                add_backward_input<<<blocks, threads>>>(grad_out, grad_input, out_rows * out_cols);

                // ✅ bias의 gradient도 계산
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, out_cols * sizeof(float));

                add_backward_bias<<<blocks, threads>>>(grad_out, grad_bias, out_rows, out_cols);
                gradients[op.param_id] = grad_bias;
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH:
                activation_backward<<<blocks, threads>>>(
                    grad_out,
                    tensors[op.output_id],  // output tensor (y)
                    grad_input,
                    out_rows,
                    out_cols,
                    op.op_type              // 3=sigmoid, 2=relu, 4=tanh
                );
                break;
            default:
                std::cerr << "[ERROR] Unsupported op_type: " << op.op_type << std::endl;
                break;
        }


        cudaDeviceSynchronize();
        gradients[op.input_id] = grad_input;
    }

    std::cout << "??run_graph_backward finished.\n";
}
