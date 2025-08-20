#pragma once

// 프로젝트의 OpType 값(전역 enum 값)과 일치해야 함
// SIGMOID/RELU/... 상수는 op_structs.cuh에서 정의된 것을 사용
inline int ge_map_act_type(int op_type) {
    switch (op_type) {
        case SIGMOID:    return ACT_SIGMOID;
        case RELU:       return ACT_RELU;
        case TANH:       return ACT_TANH;
        case LEAKY_RELU: return ACT_LEAKY;
        case ELU:        return ACT_ELU;
        case GELU:       return ACT_GELU;
        case SILU:       return ACT_SILU;
        default:         return ACT_IDENTITY;
    }
}
