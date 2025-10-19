#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef PHILOX_M4x32_A
#define PHILOX_M4x32_A 0xD2511F53u
#endif
#ifndef PHILOX_M4x32_B
#define PHILOX_M4x32_B 0xCD9E8D57u
#endif
#ifndef PHILOX_W32_A
#define PHILOX_W32_A   0x9E3779B9u
#endif
#ifndef PHILOX_W32_B
#define PHILOX_W32_B   0xBB67AE85u
#endif

struct PhiloxState {
  unsigned long long seed;   // 64-bit key
  unsigned long long offset; // 64-bit base counter
};

__device__ __forceinline__ void philox_round(uint4 &ctr, uint2 key) {
  unsigned int hi0 = __umulhi(PHILOX_M4x32_A, ctr.x);
  unsigned int hi1 = __umulhi(PHILOX_M4x32_B, ctr.z);
  unsigned int lo0 = PHILOX_M4x32_A * ctr.x;
  unsigned int lo1 = PHILOX_M4x32_B * ctr.z;
  ctr.x = hi1 ^ ctr.y ^ key.x;
  ctr.y = lo1;
  ctr.z = hi0 ^ ctr.w ^ key.y;
  ctr.w = lo0;
}
__device__ __forceinline__ void bumpkey(uint2 &key) {
  key.x += PHILOX_W32_A;
  key.y += PHILOX_W32_B;
}
__device__ __forceinline__ uint4 philox4x32_10(uint4 counter, uint2 key) {
  uint4 ctr = counter; uint2 k = key;
  #pragma unroll
  for (int i=0;i<10;++i){ philox_round(ctr,k); bumpkey(k); }
  return ctr;
}
__device__ __forceinline__ uint4 make_counter(unsigned long long base, unsigned long long elem){
  unsigned long long c = base + elem;
  uint4 ctr;
  ctr.x = (uint32_t)(c & 0xffffffffull);
  ctr.y = (uint32_t)((c>>32) & 0xffffffffull);
  ctr.z = 0u; ctr.w = 0u; return ctr;
}
__device__ __forceinline__ uint2 make_key(unsigned long long seed){
  uint2 k; k.x=(uint32_t)(seed & 0xffffffffull); k.y=(uint32_t)((seed>>32)&0xffffffffull); return k;
}
__device__ __forceinline__ float uint32_to_uniform01(uint32_t x){
  return (x >> 9) * (1.0f/8388608.0f); // 2^23
}
__device__ __forceinline__ float philox_uniform01(const PhiloxState& st, unsigned long long elem_idx){
  uint4 ctr = make_counter(st.offset, elem_idx);
  uint2 key = make_key(st.seed);
  uint4 r = philox4x32_10(ctr, key);
  return uint32_to_uniform01(r.x);
}
