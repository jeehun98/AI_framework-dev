#pragma once
#include <cuda_fp16.h>

__device__ __forceinline__ float relu_f(float v){ return v>0.f? v:0.f; }
__device__ __forceinline__ half  relu_h(half v){
  return __hgt(v, __float2half(0.f)) ? v : __float2half(0.f);
}
