
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "matmul_shared_optimized.cuh"
#include "activation.cuh"
#include "add_kernel.cuh"

#define TILE_WIDTH 16

// 디버깅용 device -> host 출력 함수 (파일 저장 포함)
void print_device_matrix_to_file(const std::string& tag, const std::string& name, float* d_ptr, int rows, int cols) {
    std::vector<float> h_data(rows * cols);
    cudaMemcpy(h_data.data(), d_ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file("debug_forward_" + tag + "_" + name + ".txt");
    if (!file) return;

    file << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            file << h_data[i * cols + j] << " ";
        file << "\n";
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

        // ✅ output 메모리 새로 할당
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

        print_device_matrix_to_file(op.output_id, "input", input, shapes[op.input_id].rows, shapes[op.input_id].cols);
        if (param) {
            print_device_matrix_to_file(op.output_id, "param", param, shapes[op.param_id].rows, shapes[op.param_id].cols);
        }

        const float* bias = nullptr;

        if (op.op_type == ADD || op.op_type == RELU || op.op_type == TANH || op.op_type == SIGMOID) {
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
        print_device_matrix_to_file(op.output_id, "output", output, rows, cols);
    }

    // ✅ 최종 출력 결과 복사
    Shape out_shape = shapes[final_output_id];
    cudaMemcpy(out_host, tensors[final_output_id], out_shape.rows * out_shape.cols * sizeof(float), cudaMemcpyDeviceToHost);
}
