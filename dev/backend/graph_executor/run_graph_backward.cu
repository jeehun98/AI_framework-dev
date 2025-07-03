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
    const std::string& final_output_id)
{
    float* grad_output = gradients[final_output_id];

    for (int i = static_cast<int>(E.size()) - 1; i >= 0; --i) {
        const OpStruct& op = E[i];

        std::cout << "[BACKWARD] op_type=" << op.op_type
                  << ", input=" << op.input_id
                  << ", param=" << op.param_id
                  << ", output=" << op.output_id << std::endl;

        float* input  = tensors[op.input_id];
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

        if (!input || !output || !grad_out) {
            std::cerr << "[NULL PTR] input/output/grad_out is nullptr at op_type " << op.op_type << std::endl;
        }

        int rows = shapes[op.output_id].rows;
        int cols = shapes[op.output_id].cols;

        if (gradients.find(op.input_id) == gradients.end() || gradients[op.input_id] == nullptr) {
            float* grad_in = nullptr;
            cudaError_t err = cudaMalloc(&grad_in, rows * cols * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "[ERROR] cudaMalloc failed for grad_input: " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            cudaMemset(grad_in, 0, rows * cols * sizeof(float));
            gradients[op.input_id] = grad_in;
        }

        float* grad_input = gradients[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end()) ? tensors[op.param_id] : nullptr;

        switch (op.op_type) {
            case MATMUL: {
                Shape in_shape = shapes[op.input_id];
                Shape w_shape  = shapes[op.param_id];

                std::cout << "[BACKWARD-MATMUL] Checking param_id: " << op.param_id << std::endl;

                if (param == nullptr) {
                    std::cerr << "[ERROR] param (" << op.param_id << ") is nullptr!" << std::endl;
                }

                if (gradients.find(op.param_id) == gradients.end() || gradients[op.param_id] == nullptr) {
                    float* grad_W;
                    cudaError_t err = cudaMalloc(&grad_W, w_shape.rows * w_shape.cols * sizeof(float));
                    if (err != cudaSuccess) {
                        std::cerr << "[ERROR] cudaMalloc failed for grad_W: " << cudaGetErrorString(err) << std::endl;
                        continue;
                    }
                    cudaMemset(grad_W, 0, w_shape.rows * w_shape.cols * sizeof(float));
                    gradients[op.param_id] = grad_W;
                    std::cout << "[ALLOC] gradients[" << op.param_id << "] allocated." << std::endl;
                }

                float* grad_W = gradients[op.param_id];

                dim3 threads(16, 16);
                dim3 grid1((in_shape.cols + 15) / 16, (in_shape.rows + 15) / 16);
                dim3 grid2((w_shape.cols + 15) / 16, (w_shape.rows + 15) / 16);

                matmul_backward_input<<<grid1, threads>>>(grad_out, param, grad_input, in_shape.rows, w_shape.cols, w_shape.rows);
                cudaError_t err1 = cudaGetLastError();
                if (err1 != cudaSuccess) {
                    std::cerr << "[KERNEL ERROR] matmul_backward_input failed: " << cudaGetErrorString(err1) << std::endl;
                }

                matmul_backward_weight<<<grid2, threads>>>(input, grad_out, grad_W, in_shape.rows, in_shape.cols, w_shape.cols);
                cudaError_t err2 = cudaGetLastError();
                if (err2 != cudaSuccess) {
                    std::cerr << "[KERNEL ERROR] matmul_backward_weight failed: " << cudaGetErrorString(err2) << std::endl;
                }

                break;
            }

            case ADD: {
                cudaMemcpy(grad_input, grad_out, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "[KERNEL ERROR] cudaMemcpy for ADD failed: " << cudaGetErrorString(err) << std::endl;
                }

                if (gradients.find(op.param_id) == gradients.end() || gradients[op.param_id] == nullptr) {
                    Shape b_shape = shapes[op.param_id];
                    float* grad_b;
                    cudaError_t err = cudaMalloc(&grad_b, b_shape.rows * b_shape.cols * sizeof(float));
                    if (err != cudaSuccess) {
                        std::cerr << "[ERROR] cudaMalloc failed for grad_b: " << cudaGetErrorString(err) << std::endl;
                        continue;
                    }
                    cudaMemset(grad_b, 0, b_shape.rows * b_shape.cols * sizeof(float));
                    gradients[op.param_id] = grad_b;
                }

                float* grad_b = gradients[op.param_id];

                int threads = 256;
                int blocks = (cols + threads - 1) / threads;
                add_backward_bias<<<blocks, threads>>>(grad_out, grad_b, rows, cols);
                cudaError_t err2 = cudaGetLastError();
                if (err2 != cudaSuccess) {
                    std::cerr << "[KERNEL ERROR] add_backward_bias failed: " << cudaGetErrorString(err2) << std::endl;
                }

                break;
            }

            case SIGMOID:
            case RELU:
            case TANH: {
                int threads = 256;
                int blocks = (rows * cols + threads - 1) / threads;
                activation_backward<<<blocks, threads>>>(grad_out, output, grad_input, rows, cols, op.op_type);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "[KERNEL ERROR] activation_backward failed: " << cudaGetErrorString(err) << std::endl;
                }
                break;
            }

            default:
                std::cerr << "[Backward] Unsupported op_type: " << op.op_type << std::endl;
                break;
        }

        float debug[2];
        cudaMemcpy(debug, grad_input, sizeof(float) * 2, cudaMemcpyDeviceToHost);
        std::cout << "Grad[" << op.input_id << "] = " << debug[0] << ", " << debug[1] << std::endl;

        cudaDeviceSynchronize();
    }

    if (gradients.find("W") != gradients.end())
        std::cout << "[CHECK] gradients[\"W\"] exists.";
}
