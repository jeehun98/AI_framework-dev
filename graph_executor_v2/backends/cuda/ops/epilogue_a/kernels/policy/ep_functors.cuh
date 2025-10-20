#pragma once
#include "ep_math.cuh"

namespace epi {

template <typename T>
struct ActNone {
  __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <typename T>
struct ActReLU {
  __device__ __forceinline__ T operator()(T x) const {
    float xf = pmath::to_f32(x);
    return pmath::from_f32<T>(pmath::relu_f(xf));
  }
};

template <typename T>
struct ActGELU {
  __device__ __forceinline__ T operator()(T x) const {
    float xf = pmath::to_f32(x);
    return pmath::from_f32<T>(pmath::gelu_f(xf));
  }
};

template <typename T, typename Act>
struct BiasAct {
  const T* bias; // length N
  Act act;
  int64_t N;
  __device__ __forceinline__ T operator()(T x, int64_t col) const {
    T xb = x;
    if (bias) {
      float xf = pmath::to_f32(xb);
      float bf = pmath::to_f32(bias[col]);
      xb = pmath::from_f32<T>(xf + bf);
    }
    return act(xb);
  }
};

} // namespace epi
