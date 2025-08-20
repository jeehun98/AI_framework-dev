#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    std::fprintf(stderr,"[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  } \
} while(0)
#endif

#ifndef CUBLAS_CHECK
#include <cublas_v2.h>
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _st = (call); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "[cuBLAS] %s:%d status=%d\n", __FILE__, __LINE__, (int)_st); \
    } \
} while(0)
#endif
