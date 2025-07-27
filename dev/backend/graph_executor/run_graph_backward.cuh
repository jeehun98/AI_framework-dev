
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "op_structs.cuh"

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size);  // ✅ 이 줄도 반드시 추가
