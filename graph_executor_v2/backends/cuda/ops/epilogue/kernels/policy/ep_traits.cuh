#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

namespace epi {

template<typename T> struct Math;

template<> struct Math<float> {
  using T=float;
  __device__ static inline T add(T a,T b){return a+b;}
  __device__ static inline T mul(T a,T b){return a*b;}
  __device__ static inline T relu(T x){return x>0.f?x:0.f;}
  __device__ static inline T gelu(T x){
    const float kA=0.7978845608028654f, kB=0.044715f;
    float t=kA*(x + kB*x*x*x);
    return 0.5f*x*(1.f+tanhf(t));
  }
  __device__ static inline T fma(T a,T b,T c){return a*b + c;}
};

template<> struct Math<half> {
  using T=half;
  __device__ static inline T add(T a,T b){return __hadd(a,b);}
  __device__ static inline T mul(T a,T b){return __hmul(a,b);}
  __device__ static inline T relu(T x){return __hgt(x,__float2half(0.f))?x:__float2half(0.f);}
  __device__ static inline T gelu(T x){
    float xf=__half2float(x);
    const float kA=0.7978845608028654f, kB=0.044715f;
    float t=kA*(xf + kB*xf*xf*xf);
    return __float2half(0.5f*xf*(1.f+tanhf(t)));
  }
  __device__ static inline T fma(T a,T b,T c){
    return __float2half(__half2float(a)*__half2float(b)+__half2float(c));
  }
};

// convert helpers
template<typename To, typename From>
__device__ inline To to(From x);
template<> __device__ inline float to<float,float>(float x){return x;}
template<> __device__ inline float to<float,half>(half x){return __half2float(x);}
template<> __device__ inline half  to<half,float>(float x){return __float2half(x);}
template<> __device__ inline half  to<half,half>(half x){return x;}

} // namespace epi
