#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "../op_structs.cuh"  // OpStruct, Shape

// Forward 연산 + Loss 계산을 수행
float run_graph_with_loss_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size);
