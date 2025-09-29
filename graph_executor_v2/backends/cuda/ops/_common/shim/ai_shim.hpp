#pragma once
// Standalone 모드: ai_core 없이 빌드할 때 쓰는 얇은 타입 정의
// BUILD_STANDALONE_OPS 가 정의되어 있을 때 활성화됨.

#ifdef BUILD_STANDALONE_OPS

#include <cstdint>
#include <array>

namespace ai {

enum class Status {
  Ok = 0,
  DeviceMismatch,
  DtypeMismatch,
  LayoutMismatch,
  TransposeNotSupported,
  ShapeMismatch,
  StrideMismatch,
  MissingInput,
  Invalid
};

using StreamHandle = void*;

enum class DType  { F32 = 0 };
enum class Layout { RowMajor = 0 };

enum class ActKind {
  None = 0,
  ReLU,
  LeakyReLU,
  GELU,
  Sigmoid,
  Tanh
};

struct TensorDesc {
  DType dtype{DType::F32};
  Layout layout{Layout::RowMajor};
  // 최대 2D만 쓰지만 여유 있게 4D 슬롯 제공
  std::array<int64_t,4> shape{0,0,0,0};
  std::array<int64_t,4> stride{0,0,0,0};
  int ndim{0};
};

struct Tensor {
  void* data{nullptr};   // device pointer (CUDA)
  TensorDesc desc{};

  // standalone에서는 CUDA만 지원 가정
  bool is_cuda() const { return true; }
};

struct GemmAttrs {
  bool trans_a{false};
  bool trans_b{false};
  ActKind act{ActKind::None};
  float leaky_slope{0.01f};
};

} // namespace ai

#else
  // 통합 빌드 경로: 원래 ai 헤더 사용
  #include "ai/dispatch.hpp"
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif
