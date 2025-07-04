// run_graph.cuh
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "op_structs.cuh"

// ✅ 최종 실행 함수: E 행렬을 기반으로 CUDA 실행
void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id);
