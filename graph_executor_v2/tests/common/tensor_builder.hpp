#pragma once
#include <vector>
#include <cstdint>
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace tbuild {

// (rows, cols) row-major, ld = cols
inline ai::Tensor make_tensor2d_f32(void* device_ptr, int64_t rows, int64_t cols){
  ai::Tensor t;
  t.data = device_ptr;
  t.device = ai::Device::CUDA;
  t.device_index = 0;
  t.desc.dtype  = ai::DType::F32;
  t.desc.layout = ai::Layout::RowMajor;
  t.desc.shape  = {rows, cols};
  t.desc.stride = {cols, 1};
  return t;
}

// (rows, cols, ld) row-major
inline ai::Tensor make_tensor2d_f32_ld(void* device_ptr, int64_t rows, int64_t cols, int64_t ld){
  ai::Tensor t;
  t.data = device_ptr;
  t.device = ai::Device::CUDA;
  t.device_index = 0;
  t.desc.dtype  = ai::DType::F32;
  t.desc.layout = ai::Layout::RowMajor;
  t.desc.shape  = {rows, cols};
  t.desc.stride = {ld, 1};
  return t;
}

// Bias per-N: (1,N) 권장
inline ai::Tensor make_bias_perN(void* device_ptr, int64_t N){
  return make_tensor2d_f32_ld(device_ptr, /*rows*/1, /*cols*/N, /*ld*/N);
}

} // namespace tbuild
