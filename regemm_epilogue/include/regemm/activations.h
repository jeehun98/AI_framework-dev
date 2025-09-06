#pragma once
namespace regemm {
__device__ __forceinline__ float act_none(float x){ return x; }
__device__ __forceinline__ float act_relu(float x){ return x > 0.f ? x : 0.f; }
} // namespace regemm
