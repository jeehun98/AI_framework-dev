#include <cuda_runtime.h>
#include <stdexcept>

#include "backends/cuda/ops/epilogue/api.hpp"
#include "backends/cuda/ops/gemm/detail/epilogue_adaptor.hpp"

namespace {
inline regemm::BiasMode to_bias_mode(ai::BiasLayout bl) {
  using BM = regemm::BiasMode;
  switch (bl) {
    case ai::BiasLayout::PerM:   return BM::PerM;
    case ai::BiasLayout::PerN:   return BM::PerN;
    case ai::BiasLayout::Scalar: return BM::Full;
    default:                     return BM::None;
  }
}
} // anon

namespace ai {

Status EpilogueFwdLaunch(const EpilogueFwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream)
{
  if (p.M <= 0 || p.N <= 0) return Status::Invalid;
  if (!p.X || !p.Y) return Status::MissingInput;
  if (p.ldX < p.N || p.ldY < p.N) return Status::StrideMismatch;
  if (a.save_z) {
    if (!p.Z) return Status::MissingOutput;
    if (p.ldZ < p.N) return Status::StrideMismatch;
  }

  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const regemm::BiasMode bm = to_bias_mode(p.bias_layout);

  auto launch = [&](auto AK_tag) {
    constexpr ai::ActKind AK = decltype(AK_tag)::value;
    switch (bm) {
      case regemm::BiasMode::PerM:
        if (a.save_z) epilogue_fwd_launch<AK, regemm::BiasMode::PerM, true >
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope, s);
        else          epilogue_fwd_launch<AK, regemm::BiasMode::PerM, false>
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0,   p.M, p.N, a.leaky_slope, s);
        break;
      case regemm::BiasMode::PerN:
        if (a.save_z) epilogue_fwd_launch<AK, regemm::BiasMode::PerN, true >
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope, s);
        else          epilogue_fwd_launch<AK, regemm::BiasMode::PerN, false>
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0,   p.M, p.N, a.leaky_slope, s);
        break;
      case regemm::BiasMode::Full:
        if (a.save_z) epilogue_fwd_launch<AK, regemm::BiasMode::Full, true >
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope, s);
        else          epilogue_fwd_launch<AK, regemm::BiasMode::Full, false>
            (p.X, p.ldX, p.Bias, p.Y, p.ldY, nullptr, 0,   p.M, p.N, a.leaky_slope, s);
        break;
      default:
        if (a.save_z) epilogue_fwd_launch<AK, regemm::BiasMode::None, true >
            (p.X, p.ldX, nullptr, p.Y, p.ldY, p.Z, p.ldZ, p.M, p.N, a.leaky_slope, s);
        else          epilogue_fwd_launch<AK, regemm::BiasMode::None, false>
            (p.X, p.ldX, nullptr, p.Y, p.ldY, nullptr, 0,   p.M, p.N, a.leaky_slope, s);
        break;
    }
  };

  switch (a.act) {
    case ActKind::ReLU:      launch(std::integral_constant<ActKind, ActKind::ReLU>{}); break;
    case ActKind::LeakyReLU: launch(std::integral_constant<ActKind, ActKind::LeakyReLU>{}); break;
    case ActKind::GELU:      launch(std::integral_constant<ActKind, ActKind::GELU>{}); break;
    case ActKind::Sigmoid:   launch(std::integral_constant<ActKind, ActKind::Sigmoid>{}); break;
    case ActKind::Tanh:      launch(std::integral_constant<ActKind, ActKind::Tanh>{}); break;
    default:                 launch(std::integral_constant<ActKind, ActKind::None>{}); break;
  }
  return Status::Ok;
}

Status EpilogueBwdLaunch(const EpilogueBwdParams& p,
                         const EpilogueAttrs& a,
                         StreamHandle stream)
{
  if (p.M <= 0 || p.N <= 0) return Status::Invalid;
  if (!p.gY || !p.Z || !p.gZ) return Status::MissingInput;
  if (p.ldgY < p.N || p.ldZ < p.N || p.ldgZ < p.N) return Status::StrideMismatch;
  if (p.gC && p.ldgC < p.N) return Status::StrideMismatch;

  const bool fuse_gC = (p.gC != nullptr) && (p.beta_for_gC != 0.f);
  const bool hasBias = (p.gBias != nullptr) && (p.bias_layout != BiasLayout::None);
  const regemm::BiasMode bm = to_bias_mode(p.bias_layout);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  auto launch = [&](auto AK_tag){
    constexpr ActKind AK = decltype(AK_tag)::value;
    #define CALL_BWD(FUSE, BMODE, HASB) \
      epilogue_bwd_launch<AK, FUSE, BMODE, HASB>( \
        p.gY, p.ldgY, p.Z, p.ldZ, p.gZ, p.ldgZ, p.M, p.N, \
        p.beta_for_gC, p.gC, p.ldgC, p.gBias, a.leaky_slope, s)

    switch (bm) {
      case regemm::BiasMode::PerM:
        if (fuse_gC) { if (hasBias) CALL_BWD(true,  regemm::BiasMode::PerM, true);
                       else          CALL_BWD(true,  regemm::BiasMode::PerM, false); }
        else         { if (hasBias) CALL_BWD(false, regemm::BiasMode::PerM, true);
                       else          CALL_BWD(false, regemm::BiasMode::PerM, false); }
        break;
      case regemm::BiasMode::PerN:
        if (fuse_gC) { if (hasBias) CALL_BWD(true,  regemm::BiasMode::PerN, true);
                       else          CALL_BWD(true,  regemm::BiasMode::PerN, false); }
        else         { if (hasBias) CALL_BWD(false, regemm::BiasMode::PerN, true);
                       else          CALL_BWD(false, regemm::BiasMode::PerN, false); }
        break;
      case regemm::BiasMode::Full:
        if (fuse_gC) { if (hasBias) CALL_BWD(true,  regemm::BiasMode::Full, true);
                       else          CALL_BWD(true,  regemm::BiasMode::Full, false); }
        else         { if (hasBias) CALL_BWD(false, regemm::BiasMode::Full, true);
                       else          CALL_BWD(false, regemm::BiasMode::Full, false); }
        break;
      default:
        if (fuse_gC) { if (hasBias) CALL_BWD(true,  regemm::BiasMode::None, true);
                       else          CALL_BWD(true,  regemm::BiasMode::None, false); }
        else         { if (hasBias) CALL_BWD(false, regemm::BiasMode::None, true);
                       else          CALL_BWD(false, regemm::BiasMode::None, false); }
        break;
    }
    #undef CALL_BWD
  };

  switch (a.act) {
    case ActKind::ReLU:      launch(std::integral_constant<ActKind, ActKind::ReLU>{}); break;
    case ActKind::LeakyReLU: launch(std::integral_constant<ActKind, ActKind::LeakyReLU>{}); break;
    case ActKind::GELU:      launch(std::integral_constant<ActKind, ActKind::GELU>{}); break;
    case ActKind::Sigmoid:   launch(std::integral_constant<ActKind, ActKind::Sigmoid>{}); break;
    case ActKind::Tanh:      launch(std::integral_constant<ActKind, ActKind::Tanh>{}); break;
    default:                 launch(std::integral_constant<ActKind, ActKind::None>{}); break;
  }
  return Status::Ok;
}

} // namespace ai
