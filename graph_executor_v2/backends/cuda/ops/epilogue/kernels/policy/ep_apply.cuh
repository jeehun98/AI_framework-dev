#pragma once
#include "../epilogue_params.cuh"
#include "ep_policy.cuh"

namespace epi {

// Params mapping
template<typename T> struct EpParamsT;
template<> struct EpParamsT<float>{ using type = EpParamsF32; };
template<> struct EpParamsT<half> { using type = EpParamsF16; };

template<typename Policy>
struct EpApply {
  using T = typename Policy::ElemT;
  using P = typename EpParamsT<T>::type;

  __device__ static inline void run(const P& p,
                                    int m, int n, int ix, int iy,
                                    const PhiloxState& st,
                                    unsigned long long elem_idx){
    T v = p.x[ix];
    // bias -> act -> dropout
    v = Policy::BiasF::template apply<T>(v, p.bias, n);
    v = Policy::ActF ::template apply<T>(v);
    if constexpr (Policy::UseDrop) {
      v = Policy::DropF::template apply<T>(v, st, elem_idx, p.p_drop, p.keep_scale);
    }
    // blend & optional residual
    Policy::BlendF::template store<T,float>(p.alpha, p.beta, v, p.y, iy);
    if constexpr (Policy::UseResid) {
      p.y[iy] = Math<T>::add(p.y[iy], p.resid[iy]);
    }
  }
};

} // namespace epi
