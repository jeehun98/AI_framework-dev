#pragma once
#include <cuda_fp16.h>
__device__ __forceinline__ float sigmoid_f(float x){ return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float tanh_f(float x){ return tanhf(x); }
__device__ __forceinline__ half  sigmoid_h(half xh){
  float x = __half2float(xh); return __float2half(1.f / (1.f + expf(-x)));
}
__device__ __forceinline__ half  tanh_h(half xh){
  return __float2half(tanhf(__half2float(xh)));
}
