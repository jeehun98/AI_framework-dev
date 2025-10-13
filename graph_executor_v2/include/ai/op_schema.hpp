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

  // NEW: Forward에서 (A*B + Bias) 를 Z로 저장할지 여부
  // - true: Z 버퍼에 pre-activation을 써두고, Y는 별도 activation pass로 생성
  // - false: 기존 fused 경로 사용 (성능 유지)
  bool     save_z{false};
};

} // namespace ai
