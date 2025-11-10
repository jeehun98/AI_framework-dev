// backends/cuda/ops/_common/shim/traits.hpp
#pragma once
#include "ai_defs.hpp"         // __host__/__device__ 가드
#include "enums.hpp"           // ActKind, BiasKind
#include "activations.hpp"     // act_apply<AK>(x, leaky)
#include "bias.hpp"            // load_bias(p,m,n)

namespace ai::cuda::shim {

// ------------------------------------------------------------
// 컴파일타임 Bias 모드 (런타임 BiasKind ↔ 컴파일타임 BiasMode 브릿지)
// ------------------------------------------------------------
enum class BiasMode : int { None = 0, PerM = 1, PerN = 2, Full = 3 };

AI_RD inline BiasMode to_bias_mode(BiasKind k) {
  switch (k) {
    case BiasKind::PerM:   return BiasMode::PerM;
    case BiasKind::PerN:   return BiasMode::PerN;
    case BiasKind::Scalar: return BiasMode::Full; // 좌표 불필요
    case BiasKind::None:
    default:               return BiasMode::None;
  }
}

// ------------------------------------------------------------
// Epilogue 정책 (α·acc + β·C + bias) → act → [dropout은 외부 적용] → D
// 템플릿 파라미터:
//   AK    : 활성화 종류 (ActKind)
//   BM    : Bias 모드   (BiasMode)
//   HasC  : C(=addend) 사용 여부 (컴파일타임 포함/제외)
//   SaveZ : pre-activation(Z) 저장 여부
//
// 요구사항(P 타입):
//   p.alpha, p.beta, p.leaky_slope, p.bias, p.bias_kind
//   (옵션) p.M, p.N  — PerM/PerN 디버그 또는 load_bias 템플릿에서 사용
// ------------------------------------------------------------
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
      float bias_j = 0.f,   // PerN 프리패치 값 (선택)
      float bias_m = 0.f    // PerM 캐시 값   (선택)
  ) {
    // 1) α·acc
    float pre = (p.alpha == 1.f) ? acc : (p.alpha * acc);

    // 2) + β·C (컴파일타임 제거 가능)
    if constexpr (HasC) {
      const float cin = C[m * ldc + n];
      pre = fmaf(p.beta, cin, pre); // p.beta==0이면 의미적으로 no-op
    }

    // 3) + bias (컴파일타임 분기)
    if constexpr (BM == BiasMode::PerN) {
      pre += bias_j;
    } else if constexpr (BM == BiasMode::PerM) {
      pre += bias_m;
    } else if constexpr (BM == BiasMode::Full) {
      pre += load_bias(p, m, n);     // Scalar 1개 경로(좌표 무시)
    } // None → no-op

    // 4) (옵션) Z 저장: ldZ==0 → ldd 사용
    if constexpr (SaveZ) {
      const int ldZ_eff = (ldZ == 0 ? ldd : ldZ);
      if (Z) Z[m * ldZ_eff + n] = pre;
    }

    // 5) act
    D[m * ldd + n] = act_apply<AK>(pre, p.leaky_slope);
  }
};

} // namespace ai::cuda::shim
