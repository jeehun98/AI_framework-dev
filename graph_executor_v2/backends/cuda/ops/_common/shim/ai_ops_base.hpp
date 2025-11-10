#pragma once
#include <cstdint>

namespace ai {

// ---------------- Activations / attrs ----------------
enum class ActKind : uint8_t { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };

struct GemmAttrs {
  bool     trans_a{false};
  bool     trans_b{false};
  ActKind  act{ActKind::None};
  bool     with_bias{false};
  float    leaky_slope{0.01f};
  bool     save_z{false}; // Z(pre-activation) 저장 의도
};

} // namespace ai
