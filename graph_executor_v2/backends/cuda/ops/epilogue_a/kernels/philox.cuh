// backends/cuda/ops/epilogue/kernels/philox.cuh
#pragma once
#include <stdint.h>

namespace epi {

// 무상태 해시 RNG (SplitMix64→u32)
__device__ __forceinline__ uint32_t splitmix32(uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return static_cast<uint32_t>(x & 0xFFFFFFFFu);
}

__device__ __forceinline__ float rand01(uint64_t seed, uint64_t idx){
  uint32_t u = splitmix32(seed ^ (idx + 0xD1342543DE82EF95ULL));
  const float scale = 1.0f / 4294967296.0f;
  return (static_cast<float>(u) + 0.5f) * scale;
}

// ep_apply 호환용 state 셰임
struct PhiloxState {
  uint64_t seed;
  __device__ __forceinline__ PhiloxState(uint64_t s=0ULL) : seed(s) {}
};

} // namespace epi
