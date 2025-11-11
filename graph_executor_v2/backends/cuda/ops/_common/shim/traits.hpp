// backends/cuda/ops/_common/shim/traits.hpp
#pragma once
#include "ai_defs.hpp"       // AI_RD
#include "enums.hpp"         // ActKind, BiasKind
#include "activations.hpp"   // act_apply<AK>(x, leaky)
#include "bias.hpp"          // load_bias(p,m,n)

namespace ai::cuda::shim {

  
// 런타임 BiasKind → 컴파일타임 정책
// Full: 좌표 독립(Scalar 1개) 바이어스
enum class BiasMode : int { None = 0, PerM = 1, PerN = 2, Full = 3 };

inline constexpr BiasMode to_bias_mode(BiasKind k) noexcept {
  switch (k) {
    case BiasKind::PerM:   return BiasMode::PerM;
    case BiasKind::PerN:   return BiasMode::PerN;
    case BiasKind::Scalar: return BiasMode::Full;
    case BiasKind::None:
    default:               return BiasMode::None;
  }
}

// Epilogue: (α·acc + β·C + bias) → act → [Z 저장 옵션]
// AK    : 활성화 종류
// BM    : Bias 정책
// HasC  : C(addend) 사용 여부
// SaveZ : pre-activation(Z) 저장 여부
template<ActKind AK, BiasMode BM, bool HasC, bool SaveZ>
struct Epilogue {
  template <typename P>
  AI_RD static void apply(
      float* __restrict__ D, int ldd,
      const float* __restrict__ C, int ldc,
      float* __restrict__ Z, int ldZ,
      const P& p,
      int m, int n,
      float acc,
      float bias_j = 0.f,   // PerN 프리패치
      float bias_m = 0.f    // PerM 캐시
  ) noexcept {
    // 1) α·acc
    float pre = (p.alpha == 1.f) ? acc : (p.alpha * acc);

    // 2) + β·C (beta==0이면 불필요한 로드 생략)
    if constexpr (HasC) {
      if (p.beta != 0.f) {
        const float cin = C[m * ldc + n];
        pre = fmaf(p.beta, cin, pre);
      }
    }

    // 3) + bias (정책별)
    if constexpr (BM == BiasMode::PerN) {
      pre += bias_j;
    } else if constexpr (BM == BiasMode::PerM) {
      pre += bias_m;
    } else if constexpr (BM == BiasMode::Full) {
      // 스칼라(1) 기준: 좌표 독립
      pre += load_bias(p, m, n);
    } // None → no-op

    // 4) Z 저장 (옵션)
    if constexpr (SaveZ) {
      const int ldZ_eff = (ldZ == 0 ? ldd : ldZ);
      if (Z) Z[m * ldZ_eff + n] = pre;
    }

    // 5) act
    D[m * ldd + n] = act_apply<AK>(pre, p.leaky_slope);
  }
};

} // namespace ai::cuda::shim
