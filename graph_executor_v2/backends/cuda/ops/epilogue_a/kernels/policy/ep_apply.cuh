// backends/cuda/ops/epilogue/kernels/policy/ep_apply.cuh
#pragma once
#include "../epilogue_params.cuh"
#include "ep_policy.cuh"
#include "ep_math.cuh"
#include "../philox.cuh"

namespace epi {

// EpParamsT: ElemT -> EpParamsF32/F16 매핑
template<typename T> struct EpParamsT;
template<> struct EpParamsT<float> { using type = EpParamsF32; };
template<> struct EpParamsT<half>  { using type = EpParamsF16; };

// 원소(벡터 인덱스) 단위 파이프라인
template<typename Policy>
struct EpApply {
  using T = typename Policy::ElemT;
  using P = typename EpParamsT<T>::type;

  // ix/iy: 벡터 단위 인덱스(바깥 정책이 계산), n_vec: Per-N bias 인덱스
  __device__ __forceinline__
  static void run(const P& p,
                  int /*m_vec*/, int n_vec, int ix, int iy,
                  const PhiloxState& st, unsigned long long elem_idx)
  {
    // 0) load
    T v = p.x[ix];

    // 1) bias
    v = Policy::BiasF::template apply<T>(v, p.bias, n_vec);

    // 2) activation
    v = Policy::ActF ::template apply<T>(v);

    // 3) dropout (inverted)
    if constexpr (Policy::UseDrop) {
      // DropF.apply는 (val, state, elem_idx, p_drop, keep_scale) 시그니처 가정
      v = Policy::DropF::template apply<T>(v, st, elem_idx, p.p_drop, p.keep_scale);
    }

    // 4) blend: y = alpha*v + beta*y
    Policy::BlendF::template store<T, float>(p.alpha, p.beta, v, p.y, iy);

    // 5) (옵션) residual: y += resid
    if constexpr (Policy::UseResid) {
      p.y[iy] = Math<T>::add(p.y[iy], p.resid ? p.resid[iy] : pmath::from_f32<T>(0.f));
    }
  }
};

} // namespace epi
