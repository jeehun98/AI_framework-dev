// kernels/philox.cuh  (간단 버전; 카운터만 필요)
#pragma once
#include <cuda_fp16.h>
struct PhiloxState { unsigned long long seed, offset; };
__device__ inline uint4 philox4x32_10(uint2 ctr, uint2 key); // 구현 생략
__device__ inline float rand_uniform01(PhiloxState& st, unsigned long long idx) {
  // idx를 counter로 사용 → (seed, offset+idx) 결정적
  uint2 ctr = make_uint2((unsigned)idx, (unsigned)(idx>>32));
  uint2 key = make_uint2((unsigned)st.seed, (unsigned)(st.seed>>32));
  uint4 r = philox4x32_10(ctr, key);
  // 한 개만 쓰는 단순화:
  return (r.x >> 9) * (1.0f / (1u<<23)); // ~[0,1)
}
