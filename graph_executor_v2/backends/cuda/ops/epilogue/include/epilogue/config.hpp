#pragma once
#include <cstdint>

namespace epilogue {

enum class ActKind : uint8_t { None, ReLU, LeakyReLU, GELU, Sigmoid, Tanh };
enum class BiasKind : uint8_t { None, PerM, PerN, Scalar };

enum class DType   : uint8_t { F16, BF16, F32 };
enum class SeedPolicy : uint8_t { Fixed, GlobalStep, TensorBased };

struct Cfg {
  float alpha{1.f};
  float beta{0.f};
  float leaky{0.f};          // LeakyReLU slope (0이면 ReLU)
  bool  save_z{false};       // pre-activation 저장 여부
  int   ldZ{0};              // 0이면 ldd로 간주

  // (옵션) Dropout/Scale-Shift
  bool  do_dropout{false};
  float keep_prob{1.f};
  uint64_t seed{0};
  SeedPolicy seed_policy{SeedPolicy::Fixed};

  const float* scale{nullptr};  // BN fold 등
  const float* shift{nullptr};
};

} // namespace epilogue
