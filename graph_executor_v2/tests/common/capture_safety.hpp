#pragma once
#include "cuda_check.hpp"
#include <cuda_runtime.h>

// 캡처 구간에서 op 호출이 "금지 호출 없음"을 보장하는지 점검
template<typename F>
inline void check_capture_safe(F&& fn){
  cudaStream_t s{};
  CUDA_CHECK(cudaStreamCreate(&s));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  fn(s);  // s 스트림으로 런처 호출
  cudaGraph_t g{};
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  CUDA_CHECK(cudaGraphDestroy(g));
  CUDA_CHECK(cudaStreamDestroy(s));
}
