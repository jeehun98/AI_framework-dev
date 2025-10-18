// api/epilogue.h

#pragma once
#include <cstdint>
#include "dtype.h"

namespace epi {

enum class ActKind   : uint8_t { None, ReLU };
enum class BiasKind  : uint8_t { None, PerN };
enum class Layout    : uint8_t { RowMajor };

struct Attrs {
  ActKind  act{ActKind::ReLU};
  BiasKind bias{BiasKind::PerN};
  float    alpha{1.f}, beta{0.f};     // y = alpha*x + beta*y
};

struct Tensors {
  void*       x;     // [M,N], input
  void*       y;     // [M,N], output (or in/out if beta!=0)
  const void* bias;  // [N], PerN
  int M, N;
  Layout x_layout{Layout::RowMajor}, y_layout{Layout::RowMajor};
  int ld_x{0}, ld_y{0};
};

struct Plan { Attrs attrs; int sm_target{0}; };
struct Status { bool ok{true}; const char* msg{""}; };

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream=nullptr);

} // namespace epi



/*
#pragma once
#include <cstdint>
#include <math.h>
#include "dtype.h"

namespace epi {

enum class ActKind   : uint8_t { None, ReLU, GELU, SiLU, Tanh, Leaky };
enum class BiasKind  : uint8_t { None, PerN, PerM, Scalar };
enum class ResidKind : uint8_t { None, Add, AddAlpha };
enum class QuantKind : uint8_t { None, FP32, FP16, BF16, Int8 };
enum class Layout    : uint8_t { RowMajor, ColMajor, Strided };

struct Attrs {
  ActKind   act{ActKind::None};
  BiasKind  bias{BiasKind::None};
  ResidKind resid{ResidKind::None};
  QuantKind quant{QuantKind::FP32};
  bool   save_z{false};
  bool   dropout{false};  float p_drop{0.f};
  bool   clamp{false};    float clamp_min{-INFINITY}, clamp_max{INFINITY};
  float  alpha{1.f}, beta{0.f};   // y = alpha*x + beta*y
  float  act_alpha{0.f};          // Leaky slope 등
};

struct Tensors {
  void*        x;   void* y;       // [M,N]
  const void*  bias{nullptr};      // PerN:(N) / PerM:(M) / Scalar:(1)
  const void*  resid{nullptr};     // [M,N] or broadcastable
  void*        z{nullptr};         // pre-act 저장 (save_z)
  const uint8_t* mask{nullptr};    // (옵션) 0/1 mask
  int M, N;
  Layout x_layout{Layout::RowMajor}, y_layout{Layout::RowMajor};
  int ld_x{0}, ld_y{0};
  // RNG
  uint64_t rng_seed{0}, rng_offset{0};
};

struct Plan {
  Attrs  attrs;
  int    sm_target{0};    // 예: 86, 89 …
};

struct Status { bool ok{true}; const char* msg{""}; };

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream=nullptr);

} // namespace epi

*/