// run_graph.cu (디버그용 전체 수정 버전)
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "matmul_shared_optimized.cuh"
#include "activation.cuh"
#include "add_kernel.cuh"

#define TILE_WIDTH 16

// 디버깅용 device -> host 출력 함수
void print_device_matrix(const std::string& name, float* d_ptr, int rows, int cols) {
    std::vector<float> h_data(rows * cols);
    cudaMemcpy(h_data.data(), d_ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < cols; ++j)
            std::cout << h_data[i * cols + j] << " ";
        std::cout << "\n";
    }
}

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id) {

    for (const auto& op : E) {
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end())
                        ? tensors[op.param_id] : nullptr;
        float* output;

        Shape out_shape;
        Shape in_shape = shapes[op.input_id];

        if (op.op_type == MATMUL && param != nullptr) {
            Shape w_shape = shapes[op.param_id];
            out_shape = {in_shape.rows, w_shape.cols};
        } else {
            out_shape = in_shape;
        }

        if (tensors.find(op.output_id) == tensors.end()) {
            float* out_ptr;
            cudaMalloc(&out_ptr, out_shape.rows * out_shape.cols * sizeof(float));
            tensors[op.output_id] = out_ptr;
            shapes[op.output_id] = out_shape;
        }

        output = tensors[op.output_id];
        int rows = out_shape.rows;
        int cols = out_shape.cols;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);
        int total = rows * cols;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        std::cout << "\n=== [CUDA EXECUTE] op_type: " << op.op_type
                  << " | input: " << op.input_id
                  << " | param: " << op.param_id
                  << " | output: " << op.output_id << " ===\n";

        print_device_matrix("input", input, shapes[op.input_id].rows, shapes[op.input_id].cols);
        if (param) {
            print_device_matrix("param", param, shapes[op.param_id].rows, shapes[op.param_id].cols);
        }

        const float* bias = nullptr;

        if (op.op_type == ADD || op.op_type == RELU || op.op_type == TANH) {
            bias = param;  // param이 bias일 경우
        }

        switch (op.op_type) {
            case MATMUL:
                matmul_shared_kernel_coalesced<<<dimGrid, dimBlock>>>(
                    input, param, output, rows, shapes[op.input_id].cols, cols);
                break;
            case ADD:
                add_kernel<<<blocks, threads>>>(input, bias, output, rows, cols);
                break;
            case SIGMOID:
                activation_sigmoid<<<blocks, threads>>>(input, bias, output, rows, cols);
                break;
            case RELU:
                activation_relu<<<blocks, threads>>>(input, bias, output, rows, cols);
                break;
            case TANH:
                activation_tanh<<<blocks, threads>>>(input, bias, output, rows, cols);
                break;
            default:
                std::cerr << "[ERROR] Unsupported op_type: " << op.op_type << std::endl;
                break;
        }


        cudaDeviceSynchronize();
        print_device_matrix("output", output, rows, cols);
    }

    Shape out_shape = shapes[final_output_id];
    cudaMemcpy(out_host, tensors[final_output_id], out_shape.rows * out_shape.cols * sizeof(float), cudaMemcpyDeviceToHost);
}
