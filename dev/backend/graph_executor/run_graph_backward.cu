#include "run_graph_backward.cuh"
#include "backward_kernels.cuh"
#include "activation_backward.cuh"
#include <cuda_runtime.h>
#include <iostream>

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)  // ✅ 추가
{
    float* grad_output = gradients[final_output_id];

    for (int i = static_cast<int>(E.size()) - 1; i >= 0; --i) {
        const OpStruct& op = E[i];

        std::cout << "[BACKWARD] op_type=" << op.op_type
                  << ", input=" << op.input_id
                  << ", param=" << op.param_id
                  << ", output=" << op.output_id << std::endl;

        float* input = tensors[op.input_id];
        float* output = tensors[op.output_id];
        float* grad_out = nullptr;

        if (gradients.find(op.output_id) != gradients.end()) {
            grad_out = gradients[op.output_id];
        } else if (op.output_id == final_output_id) {
            grad_out = grad_output;
            gradients[op.output_id] = grad_output;
        } else {
            std::cerr << "[Backward] Missing gradient for " << op.output_id << std::endl;
            continue;
        }

        int rows = shapes[op.output_id].rows;
        int cols = shapes[op.output_id].cols;
        size_t tensor_size = rows * cols;

        // ✅ 배치 단위 grad_input 메모리 할당
        if (gradients.find(op.input_id) == gradients.end() || gradients[op.input_id] == nullptr) {
            float* grad_in = nullptr;
            cudaMalloc(&grad_in, batch_size * tensor_size * sizeof(float));
            cudaMemset(grad_in, 0, batch_size * tensor_size * sizeof(float));
            gradients[op.input_id] = grad_in;
        }

        float* grad_input = gradients[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end()) ? tensors[op.param_id] : nullptr;

        switch (op.op_type) {
            case MATMUL: {
                Shape in_shape = shapes[op.input_id];
                Shape w_shape = shapes[op.param_id];

                if (gradients.find(op.param_id) == gradients.end() || gradients[op.param_id] == nullptr) {
                    float* grad_W;
                    cudaMalloc(&grad_W, w_shape.rows * w_shape.cols * sizeof(float));
                    cudaMemset(grad_W, 0, w_shape.rows * w_shape.cols * sizeof(float));
                    gradients[op.param_id] = grad_W;
                }

                float* grad_W = gradients[op.param_id];

                dim3 threads(16, 16);
                dim3 grid_input((in_shape.cols + 15) / 16, (in_shape.rows + 15) / 16);
                dim3 grid_weight((w_shape.cols + 15) / 16, (w_shape.rows + 15) / 16);

                for (int b = 0; b < batch_size; ++b) {
                    float* input_b     = input + b * in_shape.rows * in_shape.cols;
                    float* grad_out_b  = grad_out + b * in_shape.rows * w_shape.cols;
                    float* grad_input_b = grad_input + b * in_shape.rows * in_shape.cols;

                    matmul_backward_input<<<grid_input, threads>>>(
                        grad_out_b, param, grad_input_b,
                        in_shape.rows, w_shape.cols, w_shape.rows);

                    matmul_backward_weight<<<grid_weight, threads>>>(
                        input_b, grad_out_b, grad_W,
                        in_shape.rows, in_shape.cols, w_shape.cols);
                }

                break;
            }

            case ADD: {
                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b  = grad_out + b * tensor_size;
                    float* grad_input_b = grad_input + b * tensor_size;
                    cudaMemcpy(grad_input_b, grad_out_b, tensor_size * sizeof(float), cudaMemcpyDeviceToDevice);
                }

                if (gradients.find(op.param_id) == gradients.end() || gradients[op.param_id] == nullptr) {
                    Shape b_shape = shapes[op.param_id];
                    float* grad_b;
                    cudaMalloc(&grad_b, b_shape.rows * b_shape.cols * sizeof(float));
                    cudaMemset(grad_b, 0, b_shape.rows * b_shape.cols * sizeof(float));
                    gradients[op.param_id] = grad_b;
                }

                float* grad_b = gradients[op.param_id];
                int threads = 256;
                int blocks = (cols + threads - 1) / threads;

                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b = grad_out + b * tensor_size;
                    add_backward_bias<<<blocks, threads>>>(grad_out_b, grad_b, rows, cols);
                }

                break;
            }

            case FLATTEN: {
                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b = grad_out + b * tensor_size;
                    float* grad_input_b = grad_input + b * tensor_size;
                    cudaMemcpy(grad_input_b, grad_out_b, tensor_size * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH: {
                int threads = 256;
                int blocks = (tensor_size + threads - 1) / threads;

                for (int b = 0; b < batch_size; ++b) {
                    float* grad_out_b = grad_out + b * tensor_size;
                    float* output_b   = output + b * tensor_size;
                    float* grad_input_b = grad_input + b * tensor_size;

                    activation_backward<<<blocks, threads>>>(
                        grad_out_b, output_b, grad_input_b, rows, cols, op.op_type);
                }
                break;
            }

            default:
                std::cerr << "[Backward] Unsupported op_type: " << op.op_type << std::endl;
                break;
        }

        // 디버깅용 (샘플 하나만 확인)
        float debug[2];
        cudaMemcpy(debug, grad_input, sizeof(float) * 2, cudaMemcpyDeviceToHost);
        std::cout << "Grad[" << op.input_id << "] = " << debug[0] << ", " << debug[1] << std::endl;

        cudaDeviceSynchronize();
    }
}
