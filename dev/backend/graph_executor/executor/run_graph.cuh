// run_graph.cuh
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "../op_structs.cuh"

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size);  // ✅ 이 줄을 꼭 추가
