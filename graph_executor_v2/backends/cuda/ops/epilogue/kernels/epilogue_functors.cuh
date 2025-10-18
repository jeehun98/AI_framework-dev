// kernels/epilogue_functors.cuh
#pragma once
#include "epilogue_params.cuh"
#include "philox.cuh"

template <typename T> struct Vec2 { T x, y; }; // 예시용 벡터화

// dtype load/store helper (반쪽만 시범)
template <typename T>
__device__ inline T ld_ptr(const void* p, long long idx) {
  return static_cast<const T*>(p)[idx];
}
template <typename T>
__device__ inline void st_ptr(void* p, long long idx, T v) {
  static_cast<T*>(p)[idx] = v;
}

// Activation (GELU approx / ReLU만 예시)
__device__ inline float gelu_approx(float x) {
  const float k = 0.79788456f;   // sqrt(2/pi)
  return 0.5f*x*(1.f + tanhf(k*(x + 0.044715f*x*x*x)));
}

template <typename T>
__device__ inline T apply_act(T v, const EpParams& p) {
  switch (p.act) {
    case 1: return v > T(0) ? v : T(0);            // ReLU
    case 2: return (T)gelu_approx((float)v);       // GELU
    default: return v;
  }
}

// 공용 apply: (x[,y_old]) -> y_new
template <typename T>
__device__ inline T ep_apply_scalar(T x, T y_old, long long im, long long in,
                                    EpParams& P, PhiloxState& rng) {
  // 1) bias
  if (P.opmask & (1u<<0)) { // BIAS
    long long ib = im*P.sb_m + in*P.sb_n;
    x = x + ld_ptr<T>(P.bias, ib);
  }
  // 2) save_z
  if (P.opmask & (1u<<1)) { // SAVEZ
    long long iz = im*P.sz_m + in*P.sz_n;
    st_ptr<T>(P.z, iz, x);
  }
  // 3) act
  if (P.opmask & (1u<<2)) x = apply_act<T>(x, P);
  // 4) dropout
  if (P.opmask & (1u<<3)) {
    float u = rand_uniform01(rng, (unsigned long long)(im)*P.N + in);
    T keep = (u > P.p_drop) ? T(1) : T(0);
    x = x * keep * (T)P.keep_scale;
  }
  // 5) residual
  if (P.opmask & (1u<<4)) {
    long long ir = im*P.sr_m + in*P.sr_n;
    T r = ld_ptr<T>(P.resid, ir);
    if (P.resid_k == 1)      x = x + r;
    else if (P.resid_k == 2) x = x + (T)P.alpha * r; // AddAlpha
  }
  // 6) beta blend
  if (P.opmask & (1u<<5)) {
    x = (T)P.alpha * x + (T)P.beta * y_old;
  }
  // 7) clamp
  if (P.opmask & (1u<<6)) {
    float xf = (float)x;
    if (xf < P.clamp_min) xf = P.clamp_min;
    if (xf > P.clamp_max) xf = P.clamp_max;
    x = (T)xf;
  }
  // 8) quant (예시는 생략/FP 경로 유지)
  return x;
}
