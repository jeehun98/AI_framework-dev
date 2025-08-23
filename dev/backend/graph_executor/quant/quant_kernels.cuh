#pragma once
#include "quant_types.cuh"

namespace quant {

__device__ __forceinline__ int8_t clamp_s8(int v){ return (int8_t)max(-128, min(127, v)); }

__global__ void k_quantize_act_s8(const float* __restrict__ x, int n, QuantParams qp, int8_t* __restrict__ xq){
    float s = qp.scale; int z = qp.zero_point;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x){
        int qi = __float2int_rn(x[i]/s + z);
        xq[i] = clamp_s8(qi);
    }
}

__global__ void k_dequantize_act_s8(const int8_t* __restrict__ xq, int n, QuantParams qp, float* __restrict__ x){
    float s = qp.scale; int z = qp.zero_point;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x){
        x[i] = (float)((int)xq[i] - z) * s;
    }
}

// 가중치 per-channel(symmetric)
__global__ void k_quantize_w_s8_per_channel(const float* __restrict__ W, int OC, int K,
                                            const float* __restrict__ scales, int8_t* __restrict__ Wq){
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float s = scales[oc];
    for (int k = threadIdx.x; k < K; k += blockDim.x){
        int idx = oc*K + k;
        int qi = __float2int_rn(W[idx]/s);
        Wq[idx] = clamp_s8(qi);
    }
}

// B(row-major [OC,K]) -> col-major [K,OC]
__global__ void k_pack_row_to_col(const int8_t* __restrict__ Wrow, int OC, int K,
                                  int8_t* __restrict__ Wcol){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (k<K && oc<OC){
        Wcol[k*OC + oc] = Wrow[oc*K + k];
    }
}

} // namespace quant
