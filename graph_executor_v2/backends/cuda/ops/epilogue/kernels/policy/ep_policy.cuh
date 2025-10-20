#pragma once
#include "ep_functors.cuh"

namespace epi {

template<typename T,
         bool kHasBias, int kActId, bool kDropout, bool kUseResid>
struct EpPolicy {
  using ElemT = T;
  static constexpr bool HasBias  = kHasBias;
  static constexpr int  ActId    = kActId;   // 0=None,1=ReLU,2=GELU
  static constexpr bool UseDrop  = kDropout;
  static constexpr bool UseResid = kUseResid;

  using BiasF  = Bias<HasBias>;
  using ActF   = Act<ActId>;
  using DropF  = Dropout<UseDrop>;
  using ResidF = Residual<UseResid>;
  using BlendF = Blend;
};

} // namespace epi
