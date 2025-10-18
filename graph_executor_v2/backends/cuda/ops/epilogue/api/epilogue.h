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
};

struct Plan { Attrs attrs; int sm_target{0}; };

struct Status { bool ok{true}; const char* msg{""}; };

// Entry
Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream=nullptr);

} // namespace epi
