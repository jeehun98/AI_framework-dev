// backends/cuda/ops/_common/shim/ai_defs.hpp
#pragma once

// ---------- CUDA 한정자 보정 ----------
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

// ---------- 공통 인라인/한정자 매크로 ----------
#ifndef AI_INLINE
  #define AI_INLINE __forceinline__
#endif

// Host+Device 함수가 필요한 경우(예: 테스트)와 Device 전용 경로를 모두 지원
// * 기본: Device 전용 (__device__)
// * 테스트/호스트 컴파일 시: __host__ __device__
#ifndef AI_RD
  #ifdef AI_SHIM_TEST_ON_HOST
    #define AI_RD __host__ __device__ AI_INLINE
  #else
    #define AI_RD __device__ AI_INLINE
  #endif
#endif

// (선택) 편의 매크로
#ifndef AI_HD
  #define AI_HD __host__ __device__ AI_INLINE
#endif
#ifndef AI_DEV
  #define AI_DEV __device__ AI_INLINE
#endif
#ifndef AI_HOST_ONLY
  #define AI_HOST_ONLY __host__ AI_INLINE
#endif
