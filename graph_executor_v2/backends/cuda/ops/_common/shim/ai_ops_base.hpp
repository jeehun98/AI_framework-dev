// backends/cuda/ops/_common/shim/ai_ops_base.hpp
#pragma once
#include <cstdint>
#include "enums.hpp"  // ActKind 단일 소스

namespace ai::cuda::shim {

// 공통 GEMM 속성(런타임 메타)
struct GemmAttrs {
  bool    trans_a{false};
  bool    trans_b{false};
  ActKind act{ActKind::None};
  bool    with_bias{false};
  float   leaky_slope{0.01f};
  bool    save_z{false};     // Z(pre-activation) 저장 의도
  float   alpha{1.0f};       // (선택) α·acc
  float   beta{0.0f};        // (선택) β·C
};

} // namespace ai::cuda::shim
