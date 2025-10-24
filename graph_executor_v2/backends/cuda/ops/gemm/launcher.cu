// backends/cuda/ops/gemm/launcher.cu
// (FWD+BWD 통합, workspace 지원 / 정책화 디스패치 + BiasMode 런타임→컴파일타임 브릿지)
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <limits>

// NVTX 공용 shim
#include "backends/cuda/ops/_common/shim/nvtx.hpp"
#include "backends/cuda/ops/gemm/detail/nvtx_shim.h" // ← 추가 (NVTX_COLOR, NVTX_MARK 제공)

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/gemm/detail/gemm_common.hpp"
#include "backends/cuda/ops/gemm/detail/config.h"     // REGEMM_* 타일/블록 매크로
#include "backends/cuda/ops/gemm/detail/api.h"
#include "backends/cuda/ops/gemm/detail/traits.hpp"   // BiasMode / to_bias_mode / Epilogue 정책
#include "backends/cuda/ops/gemm/api.hpp"             // GemmWorkspace, GemmCudaLaunch/Backward
#include "backends/cuda/ops/epilogue/api.hpp"         // ⟵ 추가: Standalone Epilogue
 
#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr)                          \
  do {                                                    \
    ::ai::Status _st__ = (expr);                          \
    if (_st__ != ::ai::Status::Ok) return _st__;          \
  } while (0)
#endif


// 폴백 사용 플래그(정책): 필요 시 외부에서 오버라이드
#ifndef REGEMM_USE_STANDALONE_EPILOGUE_FALLBACK
#define REGEMM_USE_STANDALONE_EPILOGUE_FALLBACK 1
#endif

//
// 커널 선언(템플릿 인스턴스 디스패치용) — 정의는 kernels/*.cu에 존재
//
namespace regemm {
  // Non-EX (C 사용 여부/SaveZ 없음)
  template<int BM_, int BN_, int BK_, ActKind AK, BiasMode BM, bool HasC>
  __global__ void gemm_bias_act_f32_tiled_kernel(GemmBiasActParams p);
  void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);

  // EX (Z stash 포함)
  template<int BM_, int BN_, int BK_, ActKind AK, BiasMode BM, bool HasC, bool SaveZ>
  __global__ void gemm_bias_act_f32_tiled_kernel_ex(GemmBiasActParamsEx p);
  void launch_gemm_bias_act_f32_smoke_ex (const GemmBiasActParamsEx& p, cudaStream_t s);

  // BWD (정의는 kernels/regemm_backward.cu)
  void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);
} // namespace regemm

namespace {
using namespace ai::gemm_common;

// --- 관대 추론(1D/2D, (1,N),(M,1),(1,1) 허용) ---
inline regemm::BiasKind infer_bias_kind_fallback(const ai::Tensor* Bias, int64_t M, int64_t N) {
  using BK = regemm::BiasKind;
  if (!Bias || !Bias->data) return BK::None;
  const auto& s = Bias->desc.shape;
  int64_t numel = 1;
  for (auto v : s) numel *= v;
  if (numel <= 0) return BK::None;

  // 정확 매칭
  if (s.size()==2 && s[0]==1 && s[1]==N) return BK::PerN;
  if (s.size()==1 && s[0]==N)            return BK::PerN;
  if (s.size()==2 && s[0]==M && s[1]==1) return BK::PerM;
  if (s.size()==1 && s[0]==M)            return BK::PerM;
  if ((s.size()==2 && s[0]==1 && s[1]==1) ||
      (s.size()==1 && s[0]==1))          return BK::Scalar;

  // 느슨한 보정: numel 기준
  if (numel == N) return BK::PerN;
  if (numel == M) return BK::PerM;
  if (numel == 1) return BK::Scalar;

  return BK::None;
}

// Bias kind 최종 결정: 엄격(1D lenMN) → 실패 시 관대
inline regemm::BiasKind decide_bias_kind(const ai::Tensor* Bias, int64_t M, int64_t N) {
  auto strict = infer_bias_kind_1d_lenMN(Bias, M, N);
  if (strict != regemm::BiasKind::None) return strict;
  return infer_bias_kind_fallback(Bias, M, N);
}

// Bias 버퍼 크기/형식 검증(가능하면)
inline bool validate_bias_buffer(const ai::Tensor* Bias, int64_t M, int64_t N,
                                 regemm::BiasKind kind) {
  if (!Bias || !Bias->data || kind == regemm::BiasKind::None) return true;
  if (Bias->desc.dtype != ai::DType::F32) return false;

  const size_t need = regemm::expected_bias_elems(static_cast<int>(M), static_cast<int>(N), kind);
  if (need == 0) return false;

  size_t numel = 1;
  for (auto v : Bias->desc.shape) numel *= static_cast<size_t>(v);
  if (numel != need) {
    // (1,N)/(M,1)/(1,1) 관대 허용
    if (!(Bias->desc.shape.size()==2 &&
          ((kind==regemm::BiasKind::PerN && Bias->desc.shape[0]==1 && Bias->desc.shape[1]==N) ||
           (kind==regemm::BiasKind::PerM && Bias->desc.shape[0]==M && Bias->desc.shape[1]==1) ||
           (kind==regemm::BiasKind::Scalar && Bias->desc.shape[0]==1 && Bias->desc.shape[1]==1)))) {
      return false;
    }
  }
  return true;
}

inline bool validate_ws_lt(const ai::GemmWorkspace* ws) {
  if (!ws) return true;
  if (ws->lt_workspace && !regemm::is_workspace_aligned(ws->lt_workspace, 256)) return false;
  return true;
}

inline bool validate_ws_scratch(const ai::GemmWorkspace* ws, int64_t M, int64_t N) {
  if (!ws || !ws->scratch) return true;
  if (ws->scratch_bytes > 0) {
    const size_t need = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
    if (ws->scratch_bytes < need) return false;
  }
  return true;
}

// 타일/블록 파라미터(런처에서도 사용)
constexpr int BM  = REGEMM_TILE_M;
constexpr int BN  = REGEMM_TILE_N;
constexpr int BK  = REGEMM_TILE_K;
constexpr int TDX = REGEMM_BLOCK_TDX;
constexpr int TDY = REGEMM_BLOCK_TDY;

// === [EX 디스패치 헬퍼들] ===
// 고정된 BiasMode 인자로 직접 커널 호출
template<regemm::ActKind AK, regemm::BiasMode BMmode, bool HasC, bool SaveZ>
inline void launch_ex_cfg(const regemm::GemmBiasActParamsEx& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
  regemm::gemm_bias_act_f32_tiled_kernel_ex<BM, BN, BK, AK, BMmode, HasC, SaveZ><<<grid, block, 0, s>>>(p);
}

// 런타임 BiasMode → 컴파일타임 인스턴스 분배
template<regemm::ActKind AK, bool SaveZ>
inline void launch_ex_cfg_bm(const regemm::GemmBiasActParamsEx& p,
                             regemm::BiasMode bm,
                             cudaStream_t s) {
  constexpr bool HasC = false; // FWD에서는 C 미사용
  switch (bm) {
    case regemm::BiasMode::PerM:
      launch_ex_cfg<AK, regemm::BiasMode::PerM, HasC, SaveZ>(p, s); break;
    case regemm::BiasMode::PerN:
      launch_ex_cfg<AK, regemm::BiasMode::PerN, HasC, SaveZ>(p, s); break;
    case regemm::BiasMode::Full: // (Scalar)
      launch_ex_cfg<AK, regemm::BiasMode::Full, HasC, SaveZ>(p, s); break;
    case regemm::BiasMode::None:
    default:
      launch_ex_cfg<AK, regemm::BiasMode::None, HasC, SaveZ>(p, s); break;
  }
}

// (참고) Non-EX FWD 디스패치(현재는 사용하지 않음 — EX로 통합)
template<regemm::ActKind AK, regemm::BiasMode BMmode, bool HasC>
inline void launch_fwd_cfg(const regemm::GemmBiasActParams& p, cudaStream_t s) {
  dim3 block(TDX, TDY);
  dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
  regemm::gemm_bias_act_f32_tiled_kernel<BM, BN, BK, AK, BMmode, HasC><<<grid, block, 0, s>>>(p);
}

} // anonymous

namespace ai {

// =========================
// Forward (save_z + Lt WS 지원 / 정책화 디스패치)
// =========================
ai::Status GemmCudaLaunch(
    const Tensor& A, const Tensor& B, const Tensor* Bias /*=nullptr*/,
    Tensor& Y, const GemmAttrs& attrs,
    StreamHandle stream,
    Tensor* Z_saved /*=nullptr*/,
    const GemmWorkspace* ws /*=nullptr*/
) {
  NVTX_RANGE("gemm.fwd", NVTX_COLOR::Orange);

  // 1) 디바이스/형식/레이아웃 체크
  if (!is_cuda_f32_rowmajor(A) || !is_cuda_f32_rowmajor(B) || !is_cuda_f32_rowmajor(Y))
    return ai::Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b) return ai::Status::TransposeNotSupported;

  // 2) shape
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2)
    return ai::Status::ShapeMismatch;
  const int64_t M  = A.desc.shape[0];
  const int64_t K  = A.desc.shape[1];
  const int64_t Kb = B.desc.shape[0];
  const int64_t N  = B.desc.shape[1];
  if (K!=Kb || Y.desc.shape[0]!=M || Y.desc.shape[1]!=N) return ai::Status::ShapeMismatch;

  // 3) leading dims
  const int64_t lda = infer_ld_rowmajor_2d(A);
  const int64_t ldb = infer_ld_rowmajor_2d(B);
  const int64_t ldd = infer_ld_rowmajor_2d(Y);
  if (lda < K || ldb < N || ldd < N) return ai::Status::StrideMismatch;

  // 4) int32 범위
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldd)) {
    return ai::Status::Invalid;
  }

  // 5) Z 저장 여부 및 검증
  if (attrs.save_z && Z_saved == nullptr) return ai::Status::MissingOutput;
  const bool want_save_z = attrs.save_z && (Z_saved != nullptr);

  int   ldZ_i = 0;
  void* Z_ptr = nullptr;
  if (want_save_z) {
    if (!is_cuda_f32_rowmajor(*Z_saved)) return ai::Status::DeviceMismatch;
    if (Z_saved->desc.shape.size()!=2 ||
        Z_saved->desc.shape[0]!=M || Z_saved->desc.shape[1]!=N) {
      return ai::Status::ShapeMismatch;
    }
    const bool alias_Y = (Z_saved->data == Y.data);
    const int64_t ldZ = alias_Y ? ldd : infer_ld_rowmajor_2d(*Z_saved);
    if (ldZ < N) return ai::Status::StrideMismatch;
    if (!fits_int32(ldZ)) return ai::Status::Invalid;
    ldZ_i = static_cast<int>(ldZ);
    Z_ptr = Z_saved->data;
  }

  // 5.5) Workspace 가드(정렬/크기)
  if (!validate_ws_lt(ws)) return ai::Status::Invalid;

  // 6) regemm 파라미터 (Ex 경로 사용)
  regemm::GemmBiasActParamsEx p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  p.A   = A.data; p.lda = static_cast<int>(lda);
  p.B   = B.data; p.ldb = static_cast<int>(ldb);
  p.C   = nullptr; p.ldc = 0;                // C는 사용 안 함
  p.D   = Y.data; p.ldd = static_cast<int>(ldd);

  p.alpha = 1.0f;
  p.beta  = 0.0f;

  // ---- bias 전달 + kind 추론(엄격 → 관대) ----
  p.bias      = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind = decide_bias_kind(Bias, M, N);
  if (!validate_bias_buffer(Bias, M, N, p.bias_kind)) return ai::Status::Invalid;

  // ---- activation / leaky slope ----
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // ---- Z 저장: pre-activation을 단일 패스로 저장 ----
  p.Z           = want_save_z ? Z_ptr : nullptr;
  p.ldZ         = want_save_z ? ldZ_i : 0;   // 0이면 내부에서 ldd로 간주
  p.save_preact = want_save_z ? 1      : 0;

  // ---- Lt workspace (있으면 전달) ----
  p.lt_workspace       = ws ? ws->lt_workspace       : nullptr;
  p.lt_workspace_bytes = ws ? ws->lt_workspace_bytes : 0;

  // 7) 디스패치
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  const cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);

  // ===== 7.A 성능 우선: 기존 fused(EX) 경로 =====
  if (!REGEMM_USE_STANDALONE_EPILOGUE_FALLBACK) {
    NVTX_RANGE("gemm.fwd.ex_dispatch", NVTX_COLOR::Teal);
    if (tiny) { regemm::launch_gemm_bias_act_f32_smoke_ex(p, cs); return ai::Status::Ok; }

    const regemm::BiasMode bm = regemm::to_bias_mode(p.bias_kind);
    const bool SaveZ = want_save_z;

    switch (p.act) {
      case regemm::ActKind::ReLU:
        NVTX_MARK("ex.relu");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::ReLU,      true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::ReLU,      false>(p, bm, cs);
        break;
      case regemm::ActKind::LeakyReLU:
        NVTX_MARK("ex.leakyrelu");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::LeakyReLU, true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::LeakyReLU, false>(p, bm, cs);
        break;
      case regemm::ActKind::GELU:
        NVTX_MARK("ex.gelu");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::GELU,      true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::GELU,      false>(p, bm, cs);
        break;
      case regemm::ActKind::Sigmoid:
        NVTX_MARK("ex.sigmoid");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::Sigmoid,   true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::Sigmoid,   false>(p, bm, cs);
        break;
      case regemm::ActKind::Tanh:
        NVTX_MARK("ex.tanh");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::Tanh,      true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::Tanh,      false>(p, bm, cs);
        break;
      case regemm::ActKind::None:
      default:
        NVTX_MARK("ex.none");

        if (SaveZ) launch_ex_cfg_bm<regemm::ActKind::None,      true >(p, bm, cs);
        else       launch_ex_cfg_bm<regemm::ActKind::None,      false>(p, bm, cs);
        break;
    }
    return ai::Status::Ok;
  }

  // ===== 7.B 폴백: GEMM(=X) → Epilogue 호출 =====
  NVTX_RANGE("gemm.fwd.fallback", NVTX_COLOR::Gray);

  // (i) pre-activation만 먼저 만든다: EX 커널을 "act=None, bias=None"로 호출하여
  //     X = A*B 를 Y 또는 scratch에 쓴다. (SaveZ=false)
  void* Xbuf = Y.data;
  int   ldX  = static_cast<int>(ldd);
  bool  use_scratch = false;
  if (ws && ws->scratch && ws->scratch_bytes >= size_t(M)*size_t(N)*sizeof(float)) {
    Xbuf = ws->scratch;
    ldX  = static_cast<int>(N); // scratch는 [M,N] 연속 가정
    use_scratch = true;
  }

  // EX 파라미터를 복사해서, bias/act을 제거하고 출력 대상만 Xbuf로 바꾼다.
  regemm::GemmBiasActParamsEx p_x = p;
  p_x.D        = Xbuf;
  p_x.ldd      = ldX;
  p_x.bias     = nullptr;
  p_x.bias_kind= regemm::BiasKind::None;
  p_x.act      = regemm::ActKind::None;
  p_x.save_preact = 0;
  p_x.Z        = nullptr;
  p_x.ldZ      = 0;

  if (tiny) { 
    NVTX_MARK("fallback.tiny_gemm");

    regemm::launch_gemm_bias_act_f32_smoke_ex(p_x, cs); 
  } else {
    NVTX_MARK("fallback.main_gemm");

    launch_ex_cfg_bm<regemm::ActKind::None, false /*SaveZ*/>(p_x, regemm::BiasMode::None, cs);
  }

  // (ii) Standalone Epilogue 실행: Xbuf + Bias + Act (+Z) → Y
  {
    NVTX_RANGE("fallback.epilogue", NVTX_COLOR::Cyan);

    ai::EpilogueAttrs eattr;
    eattr.act         = attrs.act;
    eattr.leaky_slope = attrs.leaky_slope;
    eattr.save_z      = attrs.save_z;

    // regemm::BiasKind → ai::BiasLayout 매핑
    ai::BiasLayout bl = ai::BiasLayout::None;
    switch (p.bias_kind) {
      case regemm::BiasKind::PerM:   bl = ai::BiasLayout::PerM;   break;
      case regemm::BiasKind::PerN:   bl = ai::BiasLayout::PerN;   break;
      case regemm::BiasKind::Scalar: bl = ai::BiasLayout::Scalar; break;
      default:                       bl = ai::BiasLayout::None;   break;
    }

    // EpilogueFwdParams 채우기 (새 API는 raw 포인터/ld 기반)
    const float* bias_ptr = (p.bias && p.bias_kind != regemm::BiasKind::None)
                            ? reinterpret_cast<const float*>(p.bias) : nullptr;

    ai::EpilogueFwdParams ep{};
    ep.X          = reinterpret_cast<const float*>(Xbuf);
    ep.ldX        = ldX;
    ep.Bias       = bias_ptr;
    ep.bias_layout= bl;
    ep.Y          = reinterpret_cast<float*>(Y.data);
    ep.ldY        = static_cast<int>(ldd);
    ep.Z          = want_save_z ? reinterpret_cast<float*>(Z_ptr) : nullptr;
    ep.ldZ        = want_save_z ? ldZ_i : 0;
    ep.M          = static_cast<int>(M);
    ep.N          = static_cast<int>(N);

    AI_RETURN_IF_ERROR( ai::EpilogueFwdLaunch(ep, eattr, stream) );
  }

  // scratch를 썼으면 별도 해제는 상위 WS 정책(캡처-세이프)에서 관리
  return ai::Status::Ok;
}

// =========================
// Backward (원문 경로 유지) — 검증·성능 회귀 방지
// =========================
ai::Status GemmCudaBackward(
    const Tensor& A, const Tensor& B, const Tensor* C,
    const Tensor& gY, const Tensor& Z,
    Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
    const GemmAttrs& attrs,
    StreamHandle stream,
    const GemmWorkspace* ws /*=nullptr*/
) {
  NVTX_RANGE("gemm.bwd", NVTX_COLOR::Red);

  // 1) 디바이스/타입/레이아웃/transpose
  if (!is_cuda_f32_rowmajor(A) || !is_cuda_f32_rowmajor(B) ||
      !is_cuda_f32_rowmajor(gY) || !is_cuda_f32_rowmajor(Z))
    return ai::Status::DeviceMismatch;
  if (gA && !is_cuda_f32_rowmajor(*gA)) return ai::Status::DeviceMismatch;
  if (gB && !is_cuda_f32_rowmajor(*gB)) return ai::Status::DeviceMismatch;
  if (gC && !is_cuda_f32_rowmajor(*gC)) return ai::Status::DeviceMismatch;
  if (C  && !is_cuda_f32_rowmajor(*C))  return ai::Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b)   return ai::Status::TransposeNotSupported;

  // 2) shape
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 ||
      gY.desc.shape.size()!=2 || Z.desc.shape.size()!=2)
    return ai::Status::ShapeMismatch;

  const int64_t M  = A.desc.shape[0];
  const int64_t K  = A.desc.shape[1];
  const int64_t Kb = B.desc.shape[0];
  const int64_t N  = B.desc.shape[1];
  if (K != Kb) return ai::Status::ShapeMismatch;

  if (gY.desc.shape[0]!=M || gY.desc.shape[1]!=N) return ai::Status::ShapeMismatch;
  if (Z .desc.shape[0]!=M || Z .desc.shape[1]!=N) return ai::Status::ShapeMismatch;

  if (gA && (gA->desc.shape.size()!=2 || gA->desc.shape[0]!=M || gA->desc.shape[1]!=K)) return ai::Status::ShapeMismatch;
  if (gB && (gB->desc.shape.size()!=2 || gB->desc.shape[0]!=K || gB->desc.shape[1]!=N)) return ai::Status::ShapeMismatch;
  if (gC) {
    if (!C) return ai::Status::MissingInput;
    if (gC->desc.shape.size()!=2 || gC->desc.shape[0]!=M || gC->desc.shape[1]!=N) return ai::Status::ShapeMismatch;
  }

  // 3) leading dims
  const int64_t lda  = infer_ld_rowmajor_2d(A);
  const int64_t ldb  = infer_ld_rowmajor_2d(B);
  const int64_t ldgY = infer_ld_rowmajor_2d(gY);
  const int64_t ldZ  = infer_ld_rowmajor_2d(Z);
  if (lda < K || ldb < N || ldgY < N || ldZ < N) return ai::Status::StrideMismatch;

  int64_t ldgA = 0, ldgB = 0, ldgC = 0;
  if (gA) { ldgA = infer_ld_rowmajor_2d(*gA); if (ldgA < K) return ai::Status::StrideMismatch; }
  if (gB) { ldgB = infer_ld_rowmajor_2d(*gB); if (ldgB < N) return ai::Status::StrideMismatch; }
  if (gC) { ldgC = infer_ld_rowmajor_2d(*gC); if (ldgC < N) return ai::Status::StrideMismatch; }

  // int32 범위
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldgY) || !fits_int32(ldZ) ||
      (gA && !fits_int32(ldgA)) || (gB && !fits_int32(ldgB)) || (gC && !fits_int32(ldgC))) {
    return ai::Status::Invalid;
  }

  // 4) gBias kind (gBias 존재 시에만 의미)
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (gBias && gBias->data) {
    bk = decide_bias_kind(gBias, M, N);
    if (!validate_bias_buffer(gBias, M, N, bk)) return ai::Status::Invalid;
  }

  // 4.5) 캡처-세이프 dZ scratch / Lt workspace 검증
  if (!validate_ws_lt(ws)) return ai::Status::Invalid;
  if (!validate_ws_scratch(ws, M, N)) return ai::Status::Invalid;

  float* dZ = nullptr;
  if (ws && ws->scratch) {
    dZ = reinterpret_cast<float*>(ws->scratch);
  }

  // 5) 파라미터
  regemm::GemmBiasActBwdParams p{};
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

  // 6) 스케일/에필로그
  p.alpha = 1.0f;
  p.beta  = (C && gC) ? 1.0f : 0.0f;

  p.bias_kind   = bk;
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // 6.5) dZ scratch + Lt WS 전달
  p.gZ_scratch         = dZ;                                  // 외부 제공 시 malloc-free 없음
  p.ldgZ               = (dZ ? static_cast<int>(N) : 0);      // 제공 시 반드시 N
  p.lt_workspace       = ws ? ws->lt_workspace       : nullptr;
  p.lt_workspace_bytes = ws ? ws->lt_workspace_bytes : 0;

  // 7) 실행 (원문 경로 유지)
  {
    NVTX_RANGE("bwd.core", NVTX_COLOR::Magenta);
    regemm::gemm_bias_act_bwd_f32(p, reinterpret_cast<cudaStream_t>(stream));
  }
  

  return ai::Status::Ok;
}

} // namespace ai
