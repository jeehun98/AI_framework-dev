// (FWD+BWD 통합, workspace 지원 / 정책화 디스패치 + BiasMode 런타임→컴파일타임 브릿지)
#include <cuda_runtime.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/_common/shim/activations.hpp"
#include "backends/cuda/ops/_common/shim/bias.hpp"
#include "backends/cuda/ops/_common/shim/epilogue_functors.hpp"
#include "backends/cuda/ops/_common/shim/traits.hpp"          // safe_mul_nonneg, infer_bias_kind_1d_lenMN, expected_bias_elems ...
#include "backends/cuda/ops/gemm/kernels/config.hpp"            // 타일/블록 매크로
#include "backends/cuda/ops/gemm/api.hpp"                     // API + 커널 선언

// NVTX 색상 팔레트 별칭(호환)
using NVTX_COLOR = ::ai::cuda::shim::nvtx::Color;

namespace {
namespace shim = ::ai::cuda::shim;

// ──────────────── kernels 네임스페이스 심볼을 ai::cuda::shim으로 끌어오기 ────────────────
using ::ai::cuda::shim::GemmBiasActParamsEx;
using ::ai::cuda::shim::gemm_bias_act_f32_tiled_kernel_ex;
using ::ai::cuda::shim::launch_gemm_bias_act_f32_smoke_ex;
using ::ai::cuda::shim::GemmBiasActBwdParams;
using ::ai::cuda::shim::gemm_bias_act_bwd_f32;

// ──────────────── 타일/블록 상수화 ────────────────
constexpr int TBM = REGEMM_TILE_M;
constexpr int TBN = REGEMM_TILE_N;
constexpr int TBK = REGEMM_TILE_K;
constexpr int TDX = REGEMM_BLOCK_TDX;
constexpr int TDY = REGEMM_BLOCK_TDY;

// ──────────────── BiasKind → BiasMode (host 전용) ────────────────
// (traits.hpp 의 to_bias_mode 가 __host__/__device__ 가 아닐 수 있으므로 안전하게 로컬 제공)
static inline shim::BiasMode to_bias_mode_host(shim::BiasKind k) noexcept {
  using BM = shim::BiasMode;
  using BK = shim::BiasKind;
  switch (k) {
    case BK::PerM:   return BM::PerM;
    case BK::PerN:   return BM::PerN;
    case BK::Scalar: return BM::Full;
    case BK::None:
    default:         return BM::None;
  }
}

// ──────────────── BiasKind 추론/검증 ────────────────
inline shim::BiasKind infer_bias_kind_fallback(const shim::Tensor* Bias, std::int64_t M, std::int64_t N) {
  using BK = shim::BiasKind;
  if (!Bias || !Bias->data) return BK::None;

  const auto& s = Bias->desc.shape;
  std::int64_t numel = 1;
  for (auto v : s) {
    if (v < 0) return BK::None;
    numel = shim::safe_mul_nonneg(numel, v); // overflow-safe
    if (numel < 0) return BK::None;
  }
  if (numel == 0) return BK::None;

  if (s.size()==2 && s[0]==1 && s[1]==N) return BK::PerN;
  if (s.size()==1 && s[0]==N)            return BK::PerN;
  if (s.size()==2 && s[0]==M && s[1]==1) return BK::PerM;
  if (s.size()==1 && s[0]==M)            return BK::PerM;
  if ((s.size()==2 && s[0]==1 && s[1]==1) ||
      (s.size()==1 && s[0]==1))          return BK::Scalar;

  if (numel == N) return BK::PerN;
  if (numel == M) return BK::PerM;
  if (numel == 1) return BK::Scalar;
  return BK::None;
}

inline shim::BiasKind decide_bias_kind(const shim::Tensor* Bias, std::int64_t M, std::int64_t N) {
  auto strict = shim::infer_bias_kind_1d_lenMN(Bias, M, N);
  if (strict != shim::BiasKind::None) return strict;
  return infer_bias_kind_fallback(Bias, M, N);
}

inline bool validate_bias_buffer(const shim::Tensor* Bias, std::int64_t M, std::int64_t N,
                                 shim::BiasKind kind) {
  if (!Bias || !Bias->data || kind == shim::BiasKind::None) return true;
  if (Bias->desc.dtype != shim::DType::F32) return false;

  const std::size_t need = shim::expected_bias_elems(static_cast<int>(M), static_cast<int>(N), kind);
  if (need == 0) return false;

  std::size_t numel = 1;
  for (auto v : Bias->desc.shape) {
    if (v < 0) return false;
    numel *= static_cast<std::size_t>(v);
  }

  if (numel != need) {
    if (!(Bias->desc.shape.size()==2 &&
          ((kind==shim::BiasKind::PerN && Bias->desc.shape[0]==1 && Bias->desc.shape[1]==N) ||
           (kind==shim::BiasKind::PerM && Bias->desc.shape[0]==M && Bias->desc.shape[1]==1) ||
           (kind==shim::BiasKind::Scalar && Bias->desc.shape[0]==1 && Bias->desc.shape[1]==1)))) {
      return false;
    }
  }
  return true;
}

inline bool validate_ws_lt(const shim::GemmWorkspace* ws) {
  if (!ws) return true;
  if (ws->lt_workspace && !shim::is_workspace_aligned(ws->lt_workspace, 256)) return false;
  return true;
}

inline bool validate_ws_scratch(const shim::GemmWorkspace* ws, std::int64_t M, std::int64_t N) {
  if (!ws || !ws->scratch) return true;
  if (ws->scratch_bytes > 0) {
    const std::size_t need = static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * sizeof(float);
    if (ws->scratch_bytes < need) return false;
  }
  return true;
}

// ──────────────── 템플릿 런치 헬퍼 ────────────────
template<shim::ActKind AK, shim::BiasMode BMmode, bool HasC, bool SaveZ>
inline void launch_ex_cfg(const GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + TBN - 1) / TBN, (p.M + TBM - 1) / TBM);
  gemm_bias_act_f32_tiled_kernel_ex<TBM, TBN, TBK, AK, BMmode, HasC, SaveZ><<<grid, block, 0, s>>>(p);
}

template<shim::ActKind AK, bool SaveZ>
inline void launch_ex_cfg_bm(const GemmBiasActParamsEx& p, shim::BiasMode bm, cudaStream_t s) {
  constexpr bool HasC = false;
  switch (bm) {
    case shim::BiasMode::PerM: launch_ex_cfg<AK, shim::BiasMode::PerM, HasC, SaveZ>(p, s); break;
    case shim::BiasMode::PerN: launch_ex_cfg<AK, shim::BiasMode::PerN, HasC, SaveZ>(p, s); break;
    case shim::BiasMode::Full: launch_ex_cfg<AK, shim::BiasMode::Full, HasC, SaveZ>(p, s); break;
    case shim::BiasMode::None:
    default:                   launch_ex_cfg<AK, shim::BiasMode::None, HasC, SaveZ>(p, s); break;
  }
}

} // anon


// =======================
// Forward
// =======================
namespace ai::cuda::shim {

Status GemmCudaLaunch(
    const Tensor& A, const Tensor& B, const Tensor* Bias,
    Tensor& Y, const GemmAttrs& attrs,
    StreamHandle stream,
    Tensor* Z_saved,
    const GemmWorkspace* ws
) {
  NVTX_RANGE("gemm.fwd", NVTX_COLOR::Orange);

  // 1) device/type/layout
  if (!is_cuda_f32_rowmajor(A) || !is_cuda_f32_rowmajor(B) || !is_cuda_f32_rowmajor(Y))
    return Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b) return Status::TransposeNotSupported;

  // 2) shape
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2)
    return Status::ShapeMismatch;
  const std::int64_t M  = A.desc.shape[0];
  const std::int64_t K  = A.desc.shape[1];
  const std::int64_t Kb = B.desc.shape[0];
  const std::int64_t N  = B.desc.shape[1];
  if (K!=Kb || Y.desc.shape[0]!=M || Y.desc.shape[1]!=N) return Status::ShapeMismatch;

  // 3) ld
  const std::int64_t lda = infer_ld_rowmajor_2d(A);
  const std::int64_t ldb = infer_ld_rowmajor_2d(B);
  const std::int64_t ldd = infer_ld_rowmajor_2d(Y);
  if (lda < K || ldb < N || ldd < N) return Status::StrideMismatch;

  // 4) int32
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldd)) {
    return Status::Invalid;
  }

  // 5) Z 저장
  if (attrs.save_z && Z_saved == nullptr) return Status::MissingOutput;
  const bool want_save_z = attrs.save_z && (Z_saved != nullptr);

  int   ldZ_i = 0;
  void* Z_ptr = nullptr;
  if (want_save_z) {
    if (!is_cuda_f32_rowmajor(*Z_saved)) return Status::DeviceMismatch;
    if (Z_saved->desc.shape.size()!=2 || Z_saved->desc.shape[0]!=M || Z_saved->desc.shape[1]!=N)
      return Status::ShapeMismatch;
    const bool alias_Y = (Z_saved->data == Y.data);
    const std::int64_t ldZ = alias_Y ? ldd : infer_ld_rowmajor_2d(*Z_saved);
    if (ldZ < N) return Status::StrideMismatch;
    if (!fits_int32(ldZ)) return Status::Invalid;
    ldZ_i = static_cast<int>(ldZ);
    Z_ptr = Z_saved->data;
  }

  // 5.5) Workspace
  if (!::validate_ws_lt(ws))      return Status::Invalid;

  // 6) 파라미터
  GemmBiasActParamsEx p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  p.A   = A.data; p.lda = static_cast<int>(lda);
  p.B   = B.data; p.ldb = static_cast<int>(ldb);
  p.C   = nullptr; p.ldc = 0;
  p.D   = Y.data; p.ldd = static_cast<int>(ldd);

  p.alpha = 1.0f;
  p.beta  = 0.0f;

  p.bias      = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind = ::decide_bias_kind(Bias, M, N);
  if (!::validate_bias_buffer(Bias, M, N, p.bias_kind)) return Status::Invalid;

  p.act         = attrs.act;
  p.leaky_slope = attrs.leaky_slope;

  p.Z           = want_save_z ? Z_ptr : nullptr;
  p.ldZ         = want_save_z ? ldZ_i : 0;
  p.save_preact = want_save_z ? 1      : 0;

  p.lt_workspace       = ws ? ws->lt_workspace       : nullptr;
  p.lt_workspace_bytes = ws ? ws->lt_workspace_bytes : 0;

  // 7) 디스패치
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  const cudaStream_t cs = as_cuda_stream(stream);

  if (tiny) {
    launch_gemm_bias_act_f32_smoke_ex(p, cs);
    AI_CUDA_CHECK_LAUNCH();                    // ★ 런치 에러 체크
    return Status::Ok;
  }

  const BiasMode bm = ::to_bias_mode_host(p.bias_kind);
  const bool SaveZ = want_save_z;

  switch (p.act) {
    case ActKind::ReLU:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::ReLU,      true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::ReLU,      false>(p, bm, cs);
      break;
    case ActKind::LeakyReLU:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::LeakyReLU, true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::LeakyReLU, false>(p, bm, cs);
      break;
    case ActKind::GELU:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::GELU,      true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::GELU,      false>(p, bm, cs);
      break;
    case ActKind::Sigmoid:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::Sigmoid,   true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::Sigmoid,   false>(p, bm, cs);
      break;
    case ActKind::Tanh:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::Tanh,      true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::Tanh,      false>(p, bm, cs);
      break;
    case ActKind::None:
    default:
      if (SaveZ) ::launch_ex_cfg_bm<ActKind::None,      true >(p, bm, cs);
      else       ::launch_ex_cfg_bm<ActKind::None,      false>(p, bm, cs);
      break;
  }

  AI_CUDA_CHECK_LAUNCH();                      // ★ 런치 에러 체크
  return Status::Ok;
}

// =========================
// Backward
// =========================
Status GemmCudaBackward(
    const Tensor& A, const Tensor& B, const Tensor* C,
    const Tensor& gY, const Tensor& Z,
    Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
    const GemmAttrs& attrs,
    StreamHandle stream,
    const GemmWorkspace* ws
) {
  NVTX_RANGE("gemm.bwd", NVTX_COLOR::Red);

  if (!is_cuda_f32_rowmajor(A) || !is_cuda_f32_rowmajor(B) ||
      !is_cuda_f32_rowmajor(gY) || !is_cuda_f32_rowmajor(Z))
    return Status::DeviceMismatch;
  if (gA && !is_cuda_f32_rowmajor(*gA)) return Status::DeviceMismatch;
  if (gB && !is_cuda_f32_rowmajor(*gB)) return Status::DeviceMismatch;
  if (gC && !is_cuda_f32_rowmajor(*gC)) return Status::DeviceMismatch;
  if (C  && !is_cuda_f32_rowmajor(*C))  return Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b)   return Status::TransposeNotSupported;

  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 ||
      gY.desc.shape.size()!=2 || Z.desc.shape.size()!=2)
    return Status::ShapeMismatch;

  const std::int64_t M  = A.desc.shape[0];
  const std::int64_t K  = A.desc.shape[1];
  const std::int64_t Kb = B.desc.shape[0];
  const std::int64_t N  = B.desc.shape[1];
  if (K != Kb) return Status::ShapeMismatch;

  if (gY.desc.shape[0]!=M || gY.desc.shape[1]!=N) return Status::ShapeMismatch;
  if (Z .desc.shape[0]!=M || Z .desc.shape[1]!=N) return Status::ShapeMismatch;

  if (gA && (gA->desc.shape.size()!=2 || gA->desc.shape[0]!=M || gA->desc.shape[1]!=K)) return Status::ShapeMismatch;
  if (gB && (gB->desc.shape.size()!=2 || gB->desc.shape[0]!=K || gB->desc.shape[1]!=N)) return Status::ShapeMismatch;
  if (gC) {
    if (!C) return Status::MissingInput;
    if (gC->desc.shape.size()!=2 || gC->desc.shape[0]!=M || gC->desc.shape[1]!=N) return Status::ShapeMismatch;
  }

  const std::int64_t lda  = infer_ld_rowmajor_2d(A);
  const std::int64_t ldb  = infer_ld_rowmajor_2d(B);
  const std::int64_t ldgY = infer_ld_rowmajor_2d(gY);
  const std::int64_t ldZ  = infer_ld_rowmajor_2d(Z);
  if (lda < K || ldb < N || ldgY < N || ldZ < N) return Status::StrideMismatch;

  std::int64_t ldgA = 0, ldgB = 0, ldgC = 0;
  if (gA) { ldgA = infer_ld_rowmajor_2d(*gA); if (ldgA < K) return Status::StrideMismatch; }
  if (gB) { ldgB = infer_ld_rowmajor_2d(*gB); if (ldgB < N) return Status::StrideMismatch; }
  if (gC) { ldgC = infer_ld_rowmajor_2d(*gC); if (ldgC < N) return Status::StrideMismatch; }

  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldgY) || !fits_int32(ldZ) ||
      (gA && !fits_int32(ldgA)) || (gB && !fits_int32(ldgB)) || (gC && !fits_int32(ldgC))) {
    return Status::Invalid;
  }

  BiasKind bk = BiasKind::None;
  if (gBias && gBias->data) {
    bk = ::decide_bias_kind(gBias, M, N);
    if (!::validate_bias_buffer(gBias, M, N, bk)) return Status::Invalid;
  }

  if (!::validate_ws_lt(ws)) return Status::Invalid;
  if (!::validate_ws_scratch(ws, M, N)) return Status::Invalid;

  float* dZ = nullptr;
  if (ws && ws->scratch) dZ = reinterpret_cast<float*>(ws->scratch);

  GemmBiasActBwdParams p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  p.A   = A.data;  p.lda  = static_cast<int>(lda);
  p.B   = B.data;  p.ldb  = static_cast<int>(ldb);
  p.C   = C ? C->data : nullptr;
  p.ldc = C ? static_cast<int>(infer_ld_rowmajor_2d(*C)) : 0;

  p.gY  = gY.data; p.ldgY = static_cast<int>(ldgY);
  p.Z   = Z.data;  p.ldZ  = static_cast<int>(ldZ);

  p.gA  = gA ? gA->data : nullptr;  p.ldgA = gA ? static_cast<int>(ldgA) : 0;
  p.gB  = gB ? gB->data : nullptr;  p.ldgB = gB ? static_cast<int>(ldgB) : 0;
  p.gC  = gC ? gC->data : nullptr;  p.ldgC = gC ? static_cast<int>(ldgC) : 0;
  p.gBias = gBias ? gBias->data : nullptr;

  p.alpha = 1.0f;
  p.beta  = (C && gC) ? 1.0f : 0.0f;

  p.bias_kind   = bk;
  p.act         = attrs.act;
  p.leaky_slope = attrs.leaky_slope;

  p.gZ_scratch         = dZ;
  p.ldgZ               = (dZ ? static_cast<int>(N) : 0);
  p.lt_workspace       = ws ? ws->lt_workspace       : nullptr;
  p.lt_workspace_bytes = ws ? ws->lt_workspace_bytes : 0;

  {
    NVTX_RANGE("bwd.core", NVTX_COLOR::Magenta);
    gemm_bias_act_bwd_f32(p, as_cuda_stream(stream));
  }
  AI_CUDA_CHECK_LAUNCH();                // ★ 런치 에러 체크
  return Status::Ok;
}

} // ns ai::cuda::shim
