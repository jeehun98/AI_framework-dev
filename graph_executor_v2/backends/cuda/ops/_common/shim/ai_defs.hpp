// backends/cuda/ops/_common/shim/ai_defs.hpp
#pragma once

// CUDA 컴파일러가 아닐 경우 빈 매크로로 정의
#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __global__
    #define __global__
  #endif
  #ifndef __forceinline__
    #define __forceinline__ inline
  #endif
#endif
