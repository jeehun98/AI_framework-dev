#pragma once
#include "epilogue/config.hpp"

namespace epilogue {

// 호출 측에서 PerN/PerM 값을 미리 캐시해 전달하는 방식을 권장
struct BiasVals {
  float perN{0.f};  // 예: j축 프리패치
  float perM{0.f};  // 예: i축 캐시
};

template<BiasKind BK>
__device__ __forceinline__ float add_bias(float pre,
                                          const BiasVals& bv,
                                          const float* bias, // Scalar/임의형일 때만 사용
                                          int m, int n, int lda_like) {
  if constexpr (BK == BiasKind::PerN)   return pre + bv.perN;
  if constexpr (BK == BiasKind::PerM)   return pre + bv.perM;
  if constexpr (BK == BiasKind::Scalar) return pre + (bias ? bias[0] : 0.f);
  return pre;
}

} // namespace epilogue
