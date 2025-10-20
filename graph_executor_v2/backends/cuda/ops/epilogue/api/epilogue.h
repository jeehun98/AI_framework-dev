#pragma once
#include <cstdint>
#include "dtype.h"

namespace epi {

enum class ActKind : int32_t { None = 0, ReLU = 1, GELU = 2 };

struct Attrs {
  ActKind act = ActKind::None;
  float dropout_p = 0.0f;     // keep_prob = 1 - p
  uint64_t seed = 0ULL;       // Philox seed
  bool save_mask = false;     // if true, write mask_out (uint8)
};

struct Plan {
  int64_t rows = 0;       // M
  int64_t cols = 0;       // N
  int64_t ld_x = 0;       // leading stride for X (row-major: N)
  int64_t ld_y = 0;       // leading stride for Y
  int64_t ld_bias = 0;    // bias length, usually N (PerN)
  Attrs attrs{};
};

struct Tensors {
  void* x = nullptr;          // [M, N]
  void* bias = nullptr;       // [N] or [1,N], can be nullptr
  void* y = nullptr;          // [M, N] (output)
  void* mask_out = nullptr;   // [M, N] uint8 (optional)
};

/// Run epilogue: Y = Dropout( Act( X + Bias ) )
/// Returns cudaError_t
cudaError_t run(const Plan& plan, const Tensors& t, DType dtype);

} // namespace epi
