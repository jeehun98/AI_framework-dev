#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../ge/cuda_check.cuh"
#include "../op_structs.cuh"

// (선택) 워크스페이스 외부 선언 — 실제로 쓰지 않으면 제거 가능
extern std::unordered_map<std::string, void*> tensors_ws;

static inline int32_t* ensure_argmax_ws(
    std::unordered_map<std::string, float*>& tensors,
    size_t n_int32,
    const std::string& key)
{
    auto it = tensors.find(key);
    if (it != tensors.end() && it->second) {
        return reinterpret_cast<int32_t*>(it->second);
    }
    void* dptr = nullptr;
    CUDA_CHECK(cudaMalloc(&dptr, n_int32 * sizeof(int32_t)));
    tensors[key] = reinterpret_cast<float*>(dptr); // float* map 재활용
    return reinterpret_cast<int32_t*>(dptr);
}

// ✅ vector API 우선 해석: 가중치/편향은 params에 들어온다고 가정
static inline std::string resolve_A_id(const OpStruct& op) {
    if (!op.inputs.empty()) return op.inputs[0];
    return op.input_id; // legacy
}

// 🔧 중요: params[0] → inputs[1] → legacy(param_id) 순으로 해석
static inline std::string resolve_B_id(const OpStruct& op) {
    if (!op.params.empty())    return op.params[0]; // weights / bias 우선
    if (op.inputs.size() >= 2) return op.inputs[1]; // 진짜로 두 번째 data-input이 필요한 op
    return op.param_id;                               // legacy
}

// (옵션) 전체 벡터 접근 헬퍼가 필요하면 함께 사용
static inline const std::vector<std::string>& resolve_inputs(const OpStruct& op) {
    return op.inputs.empty() ? *reinterpret_cast<const std::vector<std::string>*>(
                                   &std::vector<std::string>{op.input_id}) : op.inputs;
}
static inline const std::vector<std::string>& resolve_params(const OpStruct& op) {
    return op.params.empty() ? *reinterpret_cast<const std::vector<std::string>*>(
                                   &std::vector<std::string>{op.param_id}) : op.params;
}
