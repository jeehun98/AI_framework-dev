// run_graph.cuh
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// 연산 종류
enum OpType {
    MATMUL = 0,
    ADD = 1,
    SIGMOID = 2,
    RELU = 3,
    TANH = 4,
    FLATTEN = 5  // 향후용
};

// 연산 구조체
struct OpStruct {
    int op_type;
    std::string input_id;
    std::string param_id;
    std::string output_id;
};

// 텐서 shape 정보
struct Shape {
    int rows;
    int cols;
};

// ✅ 최종 실행 함수: E 행렬을 기반으로 CUDA 실행
void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id);
