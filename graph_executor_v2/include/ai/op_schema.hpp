#pragma once
#include <cstdint>
#include <type_traits>

namespace ai {

// regemm와 1:1 매핑되도록 확장
enum class ActKind : std::uint8_t {
  None      = 0,
  ReLU      = 1,
  LeakyReLU = 2,
  GELU      = 3,
  Sigmoid   = 4,
  Tanh      = 5,
};

// ABI-safe: bool 대신 uint8_t 사용
struct GemmAttrs {
  std::uint8_t trans_a{0};    // 0 or 1
  std::uint8_t trans_b{0};    // 0 or 1
  std::uint8_t with_bias{0};  // 0 or 1
  std::uint8_t save_z{0};     // 0 or 1
  ActKind      act{ActKind::None};
  float        leaky_slope{0.01f}; // LeakyReLU 기본값

  // 편의 접근자
  bool transA()   const { return trans_a != 0; }
  bool transB()   const { return trans_b != 0; }
  bool hasBias()  const { return with_bias != 0; }
  bool saveZ()    const { return save_z != 0; }
};

static_assert(std::is_trivially_copyable<GemmAttrs>::value, "GemmAttrs must be POD/trivially copyable");

} // namespace ai
