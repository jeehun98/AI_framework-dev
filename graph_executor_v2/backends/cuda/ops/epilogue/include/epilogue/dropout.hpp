#pragma once
#include "epilogue/config.hpp"
#include <stdint.h>

// ---- CUDA 어트리뷰트/인라인 매크로 (host 빌드에서도 안전) ----
#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __forceinline__
    #define __forceinline__ inline
  #endif
#endif

namespace epilogue {

struct DropoutState {
  uint64_t seed{0};
  uint64_t subseq{0};
  uint64_t offset{0};
};

// uint32 → [0,1) 부동소수 변환(균등분포)
// 2^-32 = 1.0 / 4294967296.0f 상수 곱이 분기/정밀도에 유리
__host__ __device__ __forceinline__
float u32_to_uniform01(uint32_t x) {
  // NOTE: 0xFFFFFFFF도 1.0에 도달하지 않도록 [0,1) 보장
  constexpr float kInv2p32 = 1.0f / 4294967296.0f;
  return static_cast<float>(x) * kInv2p32;
}

// keep_prob ∈ (0,1] 가정. 경계값 방어 로직 포함.
// rnd는 Philox 등 외부 RNG가 공급하는 32-bit 정수 샘플(균등).
__host__ __device__ __forceinline__
float apply_dropout(float v, float keep_prob, /*rng*/ uint32_t rnd) {
  // 안전 가드: NaN/음수/경계 처리
  if (keep_prob >= 1.0f) return v;         // 드롭아웃 비활성
  if (keep_prob <= 0.0f) return 0.0f;      // 전부 드롭
  // mask 생성: U[0,1) < keep_prob → keep
  const float u = u32_to_uniform01(rnd);
  const float keep = (u < keep_prob) ? 1.0f : 0.0f;
  // Inverted dropout: E[y]=v 유지 위해 /keep_prob
  return (keep != 0.0f) ? (v / keep_prob) : 0.0f;
}

} // namespace epilogue
