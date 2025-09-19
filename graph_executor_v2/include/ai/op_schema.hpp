#pragma once
#include <cstdint>

namespace ai {

// regemm와 1:1 매핑되도록 확장
enum class ActKind : uint8_t {
  None      = 0,
  ReLU      = 1,
  LeakyReLU = 2,
  GELU      = 3,
  Sigmoid   = 4,
  Tanh      = 5,
};

struct GemmAttrs {
  bool     trans_a{false};
  bool     trans_b{false};
  ActKind  act{ActKind::None};
  bool     with_bias{false};
  float    leaky_slope{0.01f}; // LeakyReLU용 (regemm와 동일 기본값)
};

} // namespace ai
