// my_kernels.cu
#include "ge_v2_api.h"
#include "ge_v2_api_ex.h"
#include <cuda_runtime.h>
#include <cstdio>

// regemm headers (expect add_subdirectory(../regemm_epilogue ...) and include dir linked)
#include "regemm/api.h"

using namespace regemm;

// ── 여기서부터 매핑 함수 (공용) ─────────────────────────────
static inline regemm::ActKind to_regemm_act(ge2_act_kind_t a) {
  switch (a) {
    case GE2_ACT_NONE:       return regemm::ActKind::None;
    case GE2_ACT_RELU:       return regemm::ActKind::ReLU;
    case GE2_ACT_LEAKY_RELU: return regemm::ActKind::LeakyReLU;
    case GE2_ACT_GELU:       return regemm::ActKind::GELU;
    case GE2_ACT_SIGMOID:    return regemm::ActKind::Sigmoid;
    case GE2_ACT_TANH:       return regemm::ActKind::Tanh;
    default:                 return regemm::ActKind::None;
  }
}

static inline regemm::BiasKind to_regemm_bias(ge2_bias_kind_t b, int has_bias_flag) {
  if (!has_bias_flag) return regemm::BiasKind::None;
  switch (b) {
    case GE2_BIAS_SCALAR: return regemm::BiasKind::Scalar;
    case GE2_BIAS_PER_M:  return regemm::BiasKind::PerM;
    case GE2_BIAS_PER_N:  return regemm::BiasKind::PerN;
    default:              return regemm::BiasKind::None;
  }
}

// ---- f32 adapter: route to regemm (LEGACY) ----
extern "C" int ge2_launch_gemm_bias_act_f32(const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (!bufs || n < 4) return -1;
  auto s = reinterpret_cast<cudaStream_t>(stream_opaque);

  // legacy params are always the last pointer
  const auto* p_old = reinterpret_cast<const ge2_gemm_bias_act_params_t*>(bufs[n - 1]);
  if (!p_old) return -1;

  // bufs layout: A, B, [bias], D, params
  const float* A = reinterpret_cast<const float*>(bufs[0]);
  const float* B = reinterpret_cast<const float*>(bufs[1]);

  const float* bias = nullptr;
  int idxD = 2;
  if (p_old->has_bias) {
    if (n < 5) return -1;
    bias = reinterpret_cast<const float*>(bufs[2]);
    idxD = 3;
  }
  float* D = reinterpret_cast<float*>(bufs[idxD]);

  // Fill regemm params (row-major)
  GemmBiasActParams p{};
  p.M = p_old->M; p.N = p_old->N; p.K = p_old->K;
  p.A = const_cast<float*>(A); p.lda = p_old->K;
  p.B = const_cast<float*>(B); p.ldb = p_old->N;
  p.C = nullptr;               p.ldc = p_old->N;  // beta*C disabled
  p.D = D;                     p.ldd = p_old->N;
  p.alpha = 1.0f; p.beta = 0.0f;

  p.bias = const_cast<float*>(bias);
  p.bias_kind = bias ? BiasKind::PerN : BiasKind::Scalar;

  p.act = (p_old->act == 1 ? ActKind::ReLU : ActKind::None);

  gemm_bias_act_f32(p, s);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

// ---- fp16 tensorcore entry (placeholder) ----
extern "C" int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr* bufs, int n, void* stream) {
  (void)bufs; (void)n; (void)stream;
  // If you need this path, wire your existing cuBLASLt code here.
  return -99; // not implemented in this minimal integration
}

// ===== 확장 엔트리 =====
#include "ge_v2_api_ex.h"
// regemm/api.h는 위에서 이미 포함됨

// enum 매핑
static inline ActKind _map_act(ge2_act_kind_t a) {
  switch (a) {
    case GE2_ACT_RELU:       return ActKind::ReLU;
    case GE2_ACT_LEAKY_RELU: return ActKind::LeakyReLU;
    case GE2_ACT_GELU:       return ActKind::GELU;
    case GE2_ACT_SIGMOID:    return ActKind::Sigmoid;
    case GE2_ACT_TANH:       return ActKind::Tanh;
    case GE2_ACT_NONE:
    default:                 return ActKind::None;
  }
}
static inline BiasKind _map_bias(ge2_bias_kind_t b) {
  switch (b) {
    case GE2_BIAS_PER_M:  return BiasKind::PerM;
    case GE2_BIAS_PER_N:  return BiasKind::PerN;
    case GE2_BIAS_SCALAR:
    default:              return BiasKind::Scalar;
  }
}

// bufs 레이아웃 유틸 (Forward EX)
static inline int _fwd_idx_D(int use_C)      { return use_C ? 3 : 2; }
static inline int _fwd_idx_bias(int use_C)   { return use_C ? 4 : 3; }
static inline int _fwd_idx_Z(int use_C, int has_bias) {
  // A,B,(C),D,(bias),(Z),&px
  if (use_C)   return has_bias ? 5 : 4;
  else         return has_bias ? 4 : 3;
}

// ---- Forward(EX): Z stash 지원 ----
extern "C" int ge2_launch_gemm_bias_act_f32_ex(const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (!bufs || n < 4) return -1;
  auto s = reinterpret_cast<cudaStream_t>(stream_opaque);

  // 마지막이 params_ex
  const auto* px = reinterpret_cast<const ge2_gemm_bias_act_params_ex_t*>(bufs[n - 1]);
  if (!px) return -1;

  const float* A = reinterpret_cast<const float*>(bufs[0]); // [M,K]
  const float* B = reinterpret_cast<const float*>(bufs[1]); // [K,N]
  const float* C = nullptr;
  float* D = nullptr;
  const float* bias = nullptr;
  float* Z = nullptr;

  // bufs 파싱
  if (px->use_C) {
    // A,B,C,D,(bias),(Z),&px
    int need = 5; // A,B,C,D,&px
    if (px->has_bias) need += 1;
    if (px->save_preact) need += 1;
    if (n < need) return -1;

    C = reinterpret_cast<const float*>(bufs[2]);
    D = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_fwd_idx_D(1)])));
    if (px->has_bias) {
      bias = reinterpret_cast<const float*>(bufs[_fwd_idx_bias(1)]);
    }
    if (px->save_preact) {
      Z = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_fwd_idx_Z(1, px->has_bias)])));
    }
  } else {
    // A,B,D,(bias),(Z),&px
    int need = 4; // A,B,D,&px
    if (px->has_bias) need += 1;
    if (px->save_preact) need += 1;
    if (n < need) return -1;

    D = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_fwd_idx_D(0)])));
    if (px->has_bias) {
      bias = reinterpret_cast<const float*>(bufs[_fwd_idx_bias(0)]);
    }
    if (px->save_preact) {
      Z = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_fwd_idx_Z(0, px->has_bias)])));
    }
  }

  // 파라미터 구성 (EX)
  GemmBiasActParamsEx p{};
  p.M = px->M; p.N = px->N; p.K = px->K;

  // stride: 0/음수면 row-major 기본값
  p.lda = (px->lda > 0 ? px->lda : px->K);
  p.ldb = (px->ldb > 0 ? px->ldb : px->N);
  p.ldc = (px->ldc > 0 ? px->ldc : px->N);
  p.ldd = (px->ldd > 0 ? px->ldd : px->N);

  p.A = const_cast<float*>(A);
  p.B = const_cast<float*>(B);
  p.C = const_cast<float*>(C);
  p.D = D;

  p.alpha = (px->alpha == 0.f ? 1.f : px->alpha);
  p.beta  = px->beta;

  p.bias = const_cast<float*>(bias);
  p.bias_kind = _map_bias(px->bias_kind);
  p.act  = _map_act(px->act_kind);
  p.leaky_slope = px->leaky_slope;

  // Z stash
  p.Z = Z;
  p.ldZ = (px->ldZ > 0 ? px->ldZ : p.ldd); // 0이면 ldd로
  p.save_preact = px->save_preact;

  // 호출
  gemm_bias_act_f32_ex(p, s);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

// ---- Backward(EX): 고정 레이아웃 파싱 ----
// bufs: [A, B, C(or 0), gY, Z, gA, gB, gC(or 0), gBias(or 0), &pb]
extern "C" int ge2_launch_gemm_bias_act_bwd_f32_ex(const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (!bufs || n < 10) return -1; // 고정 레이아웃 최소 길이 체크
  auto s = reinterpret_cast<cudaStream_t>(stream_opaque);

  const auto* pb = reinterpret_cast<const ge2_gemm_bias_act_bwd_params_t*>(bufs[n - 1]);
  if (!pb) return -1;

  const float* A   = reinterpret_cast<const float*>(bufs[0]); // [M,K]
  const float* B   = reinterpret_cast<const float*>(bufs[1]); // [K,N]
  const float* C   = (bufs[2] ? reinterpret_cast<const float*>(bufs[2]) : nullptr); // optional

  const float* gY  = reinterpret_cast<const float*>(bufs[3]); // [M,N]
  const float* Z   = reinterpret_cast<const float*>(bufs[4]); // [M,N]

  float* gA        = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[5]))); // [M,K]
  float* gB        = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[6]))); // [K,N]

  float* gC        = (bufs[7] ? reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[7]))) : nullptr); // optional
  float* gBias     = (bufs[8] ? reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[8]))) : nullptr); // optional

  // 파라미터 구성 (나머지는 기존 그대로)
  GemmBiasActBwdParams p{};
  p.M = pb->M; p.N = pb->N; p.K = pb->K;

  // row-major 기본값 보정
  p.lda = (pb->lda > 0 ? pb->lda : pb->K);
  p.ldb = (pb->ldb > 0 ? pb->ldb : pb->N);
  p.ldc = (pb->ldc > 0 ? pb->ldc : pb->N);

  p.ldgY = (pb->ldgY > 0 ? pb->ldgY : pb->N);
  p.ldZ  = (pb->ldZ  > 0 ? pb->ldZ  : pb->N);

  p.ldgA = (pb->ldgA > 0 ? pb->ldgA : pb->K);
  p.ldgB = (pb->ldgB > 0 ? pb->ldgB : pb->N);
  p.ldgC = (pb->ldgC > 0 ? pb->ldgC : pb->N);

  p.A = const_cast<float*>(A);
  p.B = const_cast<float*>(B);
  p.C = const_cast<float*>(C);
  p.gY = const_cast<float*>(gY);
  p.Z  = const_cast<float*>(Z);

  p.gA = gA;
  p.gB = gB;
  p.gC = gC;
  p.gBias = gBias;

  p.alpha = pb->alpha;
  p.beta  = pb->beta;
  p.bias_kind = _map_bias(pb->bias_kind);
  p.act       = _map_act(pb->act_kind);
  p.leaky_slope = pb->leaky_slope;

  gemm_bias_act_bwd_f32(p, s);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}
