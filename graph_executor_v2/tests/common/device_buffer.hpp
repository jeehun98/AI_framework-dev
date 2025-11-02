#pragma once
#include "cuda_check.hpp"
#include <cstddef>

template<typename T>
struct DeviceBuffer {
  T* ptr{nullptr};
  size_t n{0};
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count): n(count) {
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)*n));
  }
  ~DeviceBuffer(){ if(ptr) cudaFree(ptr); }
  T* data() { return ptr; }
  const T* data() const { return ptr; }
  void h2d(const T* h) { CUDA_CHECK(cudaMemcpy(ptr, h, sizeof(T)*n, cudaMemcpyHostToDevice)); }
  void d2h(T* h) const { CUDA_CHECK(cudaMemcpy(h, ptr, sizeof(T)*n, cudaMemcpyDeviceToHost)); }
};
