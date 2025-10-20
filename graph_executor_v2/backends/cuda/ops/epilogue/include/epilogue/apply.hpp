#pragma once
#include "epilogue/config.hpp"
#include "epilogue/activations.hpp"
#include "epilogue/bias_broadcast.hpp"

namespace epilogue {

// 단일 커널 내 모놀리식 에필로그(컴파일타임 특수화 전제)
template<ActKind AK, BiasKind BK,
         bool HasC, bool SaveZ,
         typename ComputeT=float, typename StoreT=float>
struct Apply {
  __device__ __forceinline__
  static void run(
      // IO
      StoreT* __restrict__ D, int ldd,
      const StoreT* __restrict__ C, int ldc,
      StoreT* __restrict__ Z, int ldZ,
      // 좌표
      int m, int n,
      // 누산값
      ComputeT acc,
      // bias 전달(PerN/PerM 사전 캐시)
      const BiasVals& bv,
      // 상수/옵션
      const Cfg& cfg,
      // (옵션) Scalar bias/scale/shift 버퍼
      const float* __restrict__ bias_scalar = nullptr
  ) {
    // alpha*acc
    ComputeT pre = (cfg.alpha == 1.f) ? acc : (cfg.alpha * acc);

    // + beta*C
    if constexpr (HasC) {
      const ComputeT cin = static_cast<ComputeT>(C[m * ldc + n]);
      pre = fmaf(cfg.beta, cin, pre);
    }

    // + bias
    pre = add_bias<BK>(pre, bv, bias_scalar, m, n, ldd);

    // (선택) scale/shift (BN fold)
    if (cfg.scale) pre = pre * static_cast<ComputeT>(cfg.scale[0]); // 필요 시 PerC/PerN 확장
    if (cfg.shift) pre = pre + static_cast<ComputeT>(cfg.shift[0]);

    // (선택) Z 저장
    if constexpr (SaveZ) {
      const int ldZ_eff = (cfg.ldZ == 0 ? ldd : cfg.ldZ);
      if (Z) Z[m * ldZ_eff + n] = static_cast<StoreT>(pre);
    }

    // activation
    ComputeT y = act<AK>(pre, cfg.leaky);

    // (선택) dropout — 실제 사용 시 Philox 상태와 결합
    if (cfg.do_dropout) {
      // placeholder: 항등 (실투입 시 RNG 연결)
      // y = apply_dropout(y, cfg.keep_prob, rnd);
    }

    // store
    D[m * ldd + n] = static_cast<StoreT>(y);
  }
};

} // namespace epilogue
