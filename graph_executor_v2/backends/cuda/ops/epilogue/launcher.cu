#include <cuda_runtime.h>
#include <stdexcept>

#include "backends/cuda/ops/epilogue/api.hpp"
#include "backends/cuda/ops/_common/shim/traits.hpp" // BiasMode, to_bias_mode

namespace ai::cuda::shim {

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
static inline const void* pick_mask_ptr(const EpilogueFwdParams& p, bool& isFloat) {
  if (p.DropMaskF32) { isFloat = true;  return p.DropMaskF32; }
  if (p.DropMaskU8 ) { isFloat = false; return p.DropMaskU8;  }
  isFloat = false; return nullptr;
}

static inline const void* pick_mask_ptr(const EpilogueBwdParams& p, bool& isFloat) {
  if (p.DropMaskF32) { isFloat = true;  return p.DropMaskF32; }
  if (p.DropMaskU8 ) { isFloat = false; return p.DropMaskU8;  }
  isFloat = false; return nullptr;
}

static inline bool validate_dropout(const EpilogueAttrs& a) {
  if (a.dmode == DropoutMode::None) return true;
  if (a.drop_p < 0.f || a.drop_p >= 1.f) return false;
  if (a.drop_scale <= 0.f) return false;
  return true;
}

// ------------------------------------------------------------
// FWD
// ------------------------------------------------------------
Status EpilogueFwdLaunch(const EpilogueFwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream)
{
  if (p.M <= 0 || p.N <= 0) return Status::Invalid;
  if (!p.X || !p.Y)         return Status::MissingInput;
  if (p.ldX < p.N || p.ldY < p.N) return Status::StrideMismatch;
  if (a.save_z) {
    if (!p.Z)        return Status::MissingOutput;
    if (p.ldZ < p.N) return Status::StrideMismatch;
  }
  if (!validate_dropout(a)) return Status::Invalid;

  bool        mask_is_float = false;
  const void* mask_ptr = nullptr;
  if (a.dmode == DropoutMode::MaskInput) {
    mask_ptr = pick_mask_ptr(p, mask_is_float);
    if (!mask_ptr) return Status::MissingInput;
  }

  const cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);
  const BiasMode     bm = to_bias_mode(p.bias_layout); // BiasKind → BiasMode

  auto launch = [&](auto AK_tag, auto DM_tag){
    constexpr ActKind     AK = decltype(AK_tag)::value;
    constexpr DropoutMode DM = decltype(DM_tag)::value;

    switch (bm) {
      case BiasMode::PerM:
        if (a.save_z) epilogue_fwd_launch<AK, BiasMode::PerM, true,  DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        else          epilogue_fwd_launch<AK, BiasMode::PerM, false, DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        break;

      case BiasMode::PerN:
        if (a.save_z) epilogue_fwd_launch<AK, BiasMode::PerN, true,  DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        else          epilogue_fwd_launch<AK, BiasMode::PerN, false, DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        break;

      case BiasMode::Full:
        if (a.save_z) epilogue_fwd_launch<AK, BiasMode::Full, true,  DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        else          epilogue_fwd_launch<AK, BiasMode::Full, false, DM>(
            p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        break;

      default: // None
        if (a.save_z) epilogue_fwd_launch<AK, BiasMode::None, true,  DM>(
            p.X, p.ldX, nullptr, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        else          epilogue_fwd_launch<AK, BiasMode::None, false, DM>(
            p.X, p.ldX, nullptr, p.Y, p.ldY, nullptr, 0, p.M, p.N, a.leaky_slope,
            a.drop_p, a.drop_scale, mask_ptr, mask_is_float,
            a.rng_seed, a.rng_subseq, a.rng_offset, s);
        break;
    }
  };

  auto call_by_act = [&](auto DM_tag){
    switch (a.act) {
      case ActKind::ReLU:      launch(std::integral_constant<ActKind, ActKind::ReLU>{},      DM_tag); break;
      case ActKind::LeakyReLU: launch(std::integral_constant<ActKind, ActKind::LeakyReLU>{}, DM_tag); break;
      case ActKind::GELU:      launch(std::integral_constant<ActKind, ActKind::GELU>{},      DM_tag); break;
      case ActKind::Sigmoid:   launch(std::integral_constant<ActKind, ActKind::Sigmoid>{},   DM_tag); break;
      case ActKind::Tanh:      launch(std::integral_constant<ActKind, ActKind::Tanh>{},      DM_tag); break;
      default:                 launch(std::integral_constant<ActKind, ActKind::None>{},      DM_tag); break;
    }
  };

  switch (a.dmode) {
    case DropoutMode::None:      call_by_act(std::integral_constant<DropoutMode, DropoutMode::None>{}); break;
    case DropoutMode::MaskInput: call_by_act(std::integral_constant<DropoutMode, DropoutMode::MaskInput>{}); break;
    case DropoutMode::Philox:    call_by_act(std::integral_constant<DropoutMode, DropoutMode::Philox>{}); break;
  }
  return Status::Ok;
}

// ------------------------------------------------------------
// BWD
// ------------------------------------------------------------
Status EpilogueBwdLaunch(const EpilogueBwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream)
{
  if (p.M <= 0 || p.N <= 0) return Status::Invalid;
  if (!p.gY || !p.Z || !p.gZ) return Status::MissingInput;
  if (p.ldgY < p.N || p.ldZ < p.N || p.ldgZ < p.N) return Status::StrideMismatch;
  if (p.gC && p.ldgC < p.N) return Status::StrideMismatch;
  if (!validate_dropout(a)) return Status::Invalid;

  bool        mask_is_float = false;
  const void* mask_ptr = nullptr;
  if (a.dmode == DropoutMode::MaskInput) {
    mask_ptr = pick_mask_ptr(p, mask_is_float);
    if (!mask_ptr) return Status::MissingInput;
  }

  const bool     fuse_gC = (p.gC != nullptr) && (p.beta_for_gC != 0.f);
  const bool     hasBias = (p.gBias != nullptr) && (p.bias_layout != BiasKind::None);
  const BiasMode bm      = to_bias_mode(p.bias_layout); // BiasKind → BiasMode
  const cudaStream_t s   = reinterpret_cast<cudaStream_t>(stream);

  auto launch = [&](auto AK_tag, auto DM_tag){
    constexpr ActKind     AK = decltype(AK_tag)::value;
    constexpr DropoutMode DM = decltype(DM_tag)::value;

    #define CALL(FUSE, BMODE, HASB) \
      epilogue_bwd_launch<AK, FUSE, BMODE, HASB, DM>( \
        p.gY, p.ldgY, p.Z, p.ldZ, p.gZ, p.ldgZ, p.M, p.N, \
        p.beta_for_gC, p.gC, p.ldgC, p.gBias, a.leaky_slope, \
        a.drop_p, a.drop_scale, mask_ptr, mask_is_float, \
        a.rng_seed, a.rng_subseq, a.rng_offset, s)

    switch (bm) {
      case BiasMode::PerM:
        if (fuse_gC) { if (hasBias) CALL(true,  BiasMode::PerM, true);
                       else          CALL(true,  BiasMode::PerM, false); }
        else         { if (hasBias) CALL(false, BiasMode::PerM, true);
                       else          CALL(false, BiasMode::PerM, false); }
        break;

      case BiasMode::PerN:
        if (fuse_gC) { if (hasBias) CALL(true,  BiasMode::PerN, true);
                       else          CALL(true,  BiasMode::PerN, false); }
        else         { if (hasBias) CALL(false, BiasMode::PerN, true);
                       else          CALL(false, BiasMode::PerN, false); }
        break;

      case BiasMode::Full:
        if (fuse_gC) { if (hasBias) CALL(true,  BiasMode::Full, true);
                       else          CALL(true,  BiasMode::Full, false); }
        else         { if (hasBias) CALL(false, BiasMode::Full, true);
                       else          CALL(false, BiasMode::Full, false); }
        break;

      default: // None
        if (fuse_gC) { if (hasBias) CALL(true,  BiasMode::None, true);
                       else          CALL(true,  BiasMode::None, false); }
        else         { if (hasBias) CALL(false, BiasMode::None, true);
                       else          CALL(false, BiasMode::None, false); }
        break;
    }
    #undef CALL
  };

  auto call_by_act = [&](auto DM_tag){
    switch (a.act) {
      case ActKind::ReLU:      launch(std::integral_constant<ActKind, ActKind::ReLU>{},      DM_tag); break;
      case ActKind::LeakyReLU: launch(std::integral_constant<ActKind, ActKind::LeakyReLU>{}, DM_tag); break;
      case ActKind::GELU:      launch(std::integral_constant<ActKind, ActKind::GELU>{},      DM_tag); break;
      case ActKind::Sigmoid:   launch(std::integral_constant<ActKind, ActKind::Sigmoid>{},   DM_tag); break;
      case ActKind::Tanh:      launch(std::integral_constant<ActKind, ActKind::Tanh>{},      DM_tag); break;
      default:                 launch(std::integral_constant<ActKind, ActKind::None>{},      DM_tag); break;
    }
  };

  switch (a.dmode) {
    case DropoutMode::None:      call_by_act(std::integral_constant<DropoutMode, DropoutMode::None>{}); break;
    case DropoutMode::MaskInput: call_by_act(std::integral_constant<DropoutMode, DropoutMode::MaskInput>{}); break;
    case DropoutMode::Philox:    call_by_act(std::integral_constant<DropoutMode, DropoutMode::Philox>{}); break;
  }
  return Status::Ok;
}

} // namespace ai::cuda::shim
