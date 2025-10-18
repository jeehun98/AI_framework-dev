#pragma once
#include "../philox.cuh"

template <typename T>
__device__ __forceinline__ T apply_dropout(T v, const PhiloxState& st,
                                           unsigned long long elem_idx,
                                           float p_drop, float keep_scale);

template <>
__device__ __forceinline__ float apply_dropout<float>(float v, const PhiloxState& st,
                                                      unsigned long long elem_idx,
                                                      float p_drop, float keep_scale) {
  float u = philox_uniform01(st, elem_idx);
  return (u > p_drop) ? (v * keep_scale) : 0.f;
}

template <>
__device__ __forceinline__ half apply_dropout<half>(half v, const PhiloxState& st,
                                                    unsigned long long elem_idx,
                                                    float p_drop, float keep_scale) {
  float u = philox_uniform01(st, elem_idx);
  return (u > p_drop) ? __hmul(v, __float2half(keep_scale)) : __float2half(0.f);
}
