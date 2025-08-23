#pragma once
#include <unordered_map>
#include <string>
#include "../ge/cuda_check.cuh"
#include "../executor/run_graph.cuh"  // Shape 정의 사용 (rows, cols)

// 공통 출력 버퍼 보장(없으면 cudaMalloc, shapes 갱신)
inline float* ge_ensure_output(std::unordered_map<std::string, float*>& tensors,
                               std::unordered_map<std::string, Shape>& shapes,
                               const std::string& out_id,
                               const Shape& out_shape,
                               int batch_size)
{
    auto it = tensors.find(out_id);
    if (it != tensors.end() && it->second) {
        shapes[out_id] = out_shape;
        return it->second;
    }
    float* out_ptr = nullptr;
    const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&out_ptr, bytes));
    tensors[out_id] = out_ptr;
    shapes[out_id] = out_shape;
    return out_ptr;
}
