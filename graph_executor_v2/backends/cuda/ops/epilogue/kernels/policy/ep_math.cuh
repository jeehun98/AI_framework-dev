// backends/cuda/ops/epilogue/kernels/policy/ep_math.cuh
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>

namespace epi { namespace pmath {

// cast helpers
__device__ __forceinline__ float to_f32(float x){ return x; }
__device__ __forceinline__ float to_f32(half  x){ return __half2float(x); }
__device__ __forceinline__ float gelu_f(float x){
  const float c0 = 0.044715f;
  const float rsqrt2pi = 0.7978845608f;
  float u = rsqrt2pi * (x + c0 * x * x * x);
  return 0.5f * x * (1.0f + tanhf(u));
}
__device__ __forceinline__ float relu_f(float x){ return x > 0.f ? x : 0.f; }

template <typename T>
__device__ __forceinline__ T from_f32(float x);
template <>
__device__ __forceinline__ float from_f32<float>(float x){ return x; }
template <>
__device__ __forceinline__ half from_f32<half>(float x){ return __float2half(x); }

}} // namespace epi::pmath

namespace epi {
// ep_apply가 기대하는 Math<T>::add
template <typename T>
struct Math {
  __device__ __forceinline__ static T add(T a, T b){
    float fa = pmath::to_f32(a);
    float fb = pmath::to_f32(b);
    return pmath::from_f32<T>(fa + fb);
  }
};
} // namespace epi
