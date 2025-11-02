#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA Error: ")+cudaGetErrorString(_err)); \
    } \
  } while(0)
