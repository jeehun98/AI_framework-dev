#pragma once
#include "api.h"
#include "activations.h"
#include "bias.h"

namespace regemm {

// ---------------------------------------------------------------------
// 컴파일타임 Bias 모드(PerN/PerM/Full(=Scalar)/None)
//  - 런타임 BiasKind → 컴파일타임 BiasMode 브릿지(to_bias_mode) 제공
//  - Full: Scalar(한 개) 케이스를 좌표 계산 없이 처리하기 위한 모드명
// ---------------------------------------------------------------------
enum class BiasMode : int { None = 0, PerM = 1, PerN = 2, Full = 3 };

__host__ __device__ __forceinline__
BiasMode to_bias_mode(BiasKind k) {
  switch (k) {
    case BiasKind::PerM:   return BiasMode::PerM;
    case BiasKind::PerN:   return BiasMode::PerN;
    case BiasKind::Scalar: return BiasMode::Full; // 좌표 필요 없음
    case BiasKind::None:
    default:               return BiasMode::None;
  }
}

// ---------------------------------------------------------------------
// Epilogue 정책
//   pre = alpha*acc (+ beta*C) (+ bias) [ + Z(save) ]
//   y   = act(pre)
//   D[m, n] = y
//
// 템플릿 파라미터:
//   AK     : 활성화 종류 (ActKind)
//   BM     : Bias 모드   (BiasMode)
//   HasC   : C(=addend) 사용 여부 (beta*C 경로가 컴파일타임 포함/제외)
//   SaveZ  : pre-activation(Z) 저장 여부
//
// 사용 팁:
//  - PerN/PerM 최적화를 위해 호출 측에서 bias_j(PerN prefetch), bias_m(PerM cache)을 전달
//  - C를 사용하지 않는 대부분의 FWD 경로에서는 HasC=false로 인스턴스화하면
//    C 로드/주소계산/분기가 컴파일 타임에 제거됨.
// ---------------------------------------------------------------------
template<ActKind AK, BiasMode BM, bool HasC, bool SaveZ>
struct Epilogue {
  template <typename P>
  __device__ __forceinline__
  static void apply(
      float* __restrict__ D, int ldd,
      const float* __restrict__ C, int ldc,
      float* __restrict__ Z, int ldZ,
      const P& p,
      int m, int n,
      float acc,
      float bias_j = 0.f,   // PerN 프리패치 값
      float bias_m = 0.f    // PerM 캐시 값
  ) {
    // alpha*acc
    float pre = (p.alpha == 1.f) ? acc : (p.alpha * acc);

    // + beta*C (컴파일타임 옵션)
    if constexpr (HasC) {
      const float cin = C[m * ldc + n];
      pre = fmaf(p.beta, cin, pre);
    }

    // + bias (컴파일타임 정책)
    if constexpr (BM == BiasMode::PerN) {
      pre += bias_j;
    } else if constexpr (BM == BiasMode::PerM) {
      pre += bias_m;
    } else if constexpr (BM == BiasMode::Full) {
      // Scalar 1개 → 좌표 무관
      pre += load_bias(p, m, n);
    } // None → no-op

    // (옵션) Z 저장: ldZ==0이면 ldd를 사용
    if constexpr (SaveZ) {
      const int ldZ_eff = (ldZ == 0 ? ldd : ldZ);
      if (Z) Z[m * ldZ_eff + n] = pre;
    }

    // 활성화 적용 후 저장
    D[m * ldd + n] = act_apply<AK>(pre, p.leaky_slope);
  }
};

} // namespace regemm
