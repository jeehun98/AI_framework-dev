#pragma once
#include <cstdint>
#include "dtype.h"

namespace epi {

enum class ActKind   : uint8_t { None=0, ReLU=1, GELU=2 };
enum class BiasKind  : uint8_t { None=0, PerN=1 };
enum class Layout    : uint8_t { RowMajor=0 /*, ColMajor, Strided*/ };

struct Attrs {
  ActKind  act{ActKind::ReLU};
  BiasKind bias{BiasKind::PerN};
  float    alpha{1.f}, beta{0.f};
  bool     dropout{false};
  float    p_drop{0.f}; // 0~1
};

struct Tensors {
  void*   x;     // [M,N]
  void*   y;     // [M,N]
  const void* bias; // [N] (PerN) or nullptr
  int M, N;
  Layout x_layout{Layout::RowMajor}, y_layout{Layout::RowMajor};
  int ld_x{0}, ld_y{0};
  // RNG (graph-capture safe)
  uint64_t rng_seed{0}, rng_offset{0};

  
  // NEW: optional residual pointer (e.g., h_prev for GRU, or external residual)
  const void* resid{nullptr};
};

enum class LayerRecipe : uint8_t {
  None = 0,
  // Transformer Feed-Forward epilogue:
  // y = alpha * Dropout(GELU(x + bias)) + beta * y + resid
  FFN_GELU_Dropout_Residual = 1,
  CNN_Conv_Bias_ReLU        = 2,   // NEW
  GRU3_Gates                = 3    // NEW: x=[z|r|n], y=h_t, resid=h_prev
};

struct Plan { 
  Attrs attrs; 
  int sm_target{0}; 
  LayerRecipe recipe{LayerRecipe::None};   // NEW

};

struct Status { bool ok{true}; const char* msg{""}; };

// Entry
Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream=nullptr);


// NEW: 레시피용 엔트리 (기존 run은 그대로 유지)
Status run_layer(const Plan& plan, const Tensors& ts,
                 DType xdt, DType ydt, DType bdt, void* stream=nullptr);

} // namespace epi          
