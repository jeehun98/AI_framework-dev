#pragma once
#include "../philox.cuh"
#include "ep_traits.cuh"

namespace epi {

// Bias
template<bool kHasBias> struct Bias {
  template<typename T>
  __device__ static inline T apply(T v, const T* bias, int n) {
    if constexpr (kHasBias) return Math<T>::add(v, bias[n]);
    else return v;
  }
};

// Activation: 0=None,1=ReLU,2=GELU
template<int ActId> struct Act;
template<> struct Act<0>{ template<typename T>
  __device__ static inline T apply(T x){return x;} };
template<> struct Act<1>{ template<typename T>
  __device__ static inline T apply(T x){return Math<T>::relu(x);} };
template<> struct Act<2>{ template<typename T>
  __device__ static inline T apply(T x){return Math<T>::gelu(x);} };

// Dropout
template<bool kDrop> struct Dropout {
  template<typename T>
  __device__ static inline T apply(T v, const PhiloxState& st,
                                   unsigned long long elem_idx,
                                   float p_drop, float keep) {
    if constexpr (!kDrop) return v;
    float u = philox_uniform01(st, elem_idx);
    return (u > p_drop) ? Math<T>::mul(v, to<T,float>(keep)) : to<T,float>(0.f);
  }
};

// Residual
template<bool kUse> struct Residual {
  template<typename T>
  __device__ static inline T add(T v, const T* resid, int iy) {
    if constexpr (kUse) return Math<T>::add(v, resid[iy]);
    else return v;
  }
};

// Blend y = alpha*v + beta*yold
struct Blend {
  template<typename T, typename TS>
  __device__ static inline void store(TS alpha, TS beta, T v, T* y, int iy) {
    if (beta!=TS(0)) y[iy] = Math<T>::fma(to<T,TS>(alpha), v, Math<T>::mul(to<T,TS>(beta), y[iy]));
    else             y[iy] = Math<T>::mul(to<T,TS>(alpha), v);
  }
};

} // namespace epi
