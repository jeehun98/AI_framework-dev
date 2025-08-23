#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../ge/cuda_check.cuh"

// tensor_id -> device pointer 저장한다고 가정
extern std::unordered_map<std::string, void*> tensors_ws;

static inline int32_t* ensure_argmax_ws(
    std::unordered_map<std::string, float*>& tensors,
    size_t n_int32,
    const std::string& key)
{
    auto it = tensors.find(key);
    if (it != tensors.end() && it->second) {
        // 이미 캐시되어 있음
        return reinterpret_cast<int32_t*>(it->second);
    }

    void* dptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dptr, n_int32 * sizeof(int32_t)));
    // float* 맵을 재활용하기 위해 캐스팅해서 저장
    tensors[key] = reinterpret_cast<float*>(dptr);
    return reinterpret_cast<int32_t*>(dptr);
}