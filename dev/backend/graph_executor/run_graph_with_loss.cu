#include "run_graph_with_loss.cuh"
#include "run_graph.cuh"  // 기존 forward 계산 함수
#include "loss_kernels.cuh"  // MSE, BCE 등 CUDA 커널

#include <iostream>

float run_graph_with_loss_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size)
{
    // 1. 계산 그래프 수행
    float dummy_output = 0.0f;
    run_graph_cuda(E, tensors, shapes, &dummy_output, final_output_id, batch_size);

    // 2. 출력 텐서 및 정답 텐서 접근
    float* y_pred = tensors[final_output_id];
    float* y_true = tensors[label_tensor_id];
    Shape shape = shapes[final_output_id];

    int total_elements = shape.rows * shape.cols;

    float loss_value = 0.0f;

    if (loss_type == "mse") {
        loss_value = compute_mse_loss_cuda(y_true, y_pred, total_elements);
    } else if (loss_type == "binary_crossentropy") {
        loss_value = compute_bce_loss_cuda(y_true, y_pred, total_elements);
    } else {
        std::cerr << "[ERROR] Unsupported loss type: " << loss_type << std::endl;
    }

    return loss_value;
}
