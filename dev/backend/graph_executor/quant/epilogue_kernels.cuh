#pragma once
#include "quant_types.cuh"

namespace quant {

// ACC[s32] → fp32, per-out-channel scale(sw[n])와 입력 scale(sx) 합성, bias 추가, 선택적 ReLU
__global__ void k_epilogue_dequant_bias_relu(const int32_t* __restrict__ ACC,
                                             float* __restrict__ Y, int M, int N,
                                             const float* __restrict__ sw, float sx,
                                             const float* __restrict__ bias, bool relu){
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m>=M || n>=N) return;
    float y = (float)ACC[m*N + n] * (sw[n] * sx) + (bias ? bias[n] : 0.0f);
    if (relu) y = fmaxf(0.f, y);
    Y[m*N + n] = y;
}

} // namespace quant
