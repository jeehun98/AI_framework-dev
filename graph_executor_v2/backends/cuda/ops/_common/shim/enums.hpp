// backends/cuda/ops/_common/shim/enums.hpp
#pragma once
#include <cstdint>

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Activation / Bias kinds (공통 enum 정의)
// ------------------------------------------------------------
enum class ActKind : int {
  None = 0,
  ReLU = 1,
  LeakyReLU = 2,
  GELU = 3,
  Sigmoid = 4,
  Tanh = 5
};

enum class BiasKind : int {
  None   = 0,
  Scalar = 1,
  PerM   = 2,
  PerN   = 3
};

enum class DropoutMode : std::uint8_t {
  None      = 0,
  MaskInput = 1,   // 외부에서 0/1 mask 제공
  Philox    = 2    // Philox RNG로 내부에서 생성
};

// ------------------------------------------------------------
// ABI 고정 확인
// ------------------------------------------------------------
static_assert(static_cast<int>(ActKind::ReLU)    == 1, "ActKind ABI changed");
static_assert(static_cast<int>(BiasKind::Scalar) == 1, "BiasKind ABI changed");

} // namespace ai::cuda::shim
