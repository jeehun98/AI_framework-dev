#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cstdio>

// regemm headers (expect add_subdirectory(../regemm_epilogue ...) and include dir linked)
#include "regemm/api.h"

using namespace regemm;

// ---- f32 adapter: route to regemm ----
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
#include "regemm/api.h"
using namespace regemm;

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

// bufs 레이아웃 유틸
static inline int _index_D(int use_C) { return use_C ? 3 : 2; }
static inline int _index_bias(int use_C) { return use_C ? 4 : 3; }

// 확장 진입점
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

  // bufs 파싱
  if (px->use_C) {
    if (n < 6 && px->has_bias) return -1; // A,B,C,D,bias,params → 최소 6
    if (n < 5 && !px->has_bias) return -1; // A,B,C,D,params → 최소 5
    C = reinterpret_cast<const float*>(bufs[2]);
    D = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_index_D(1)])));
    if (px->has_bias) {
      bias = reinterpret_cast<const float*>(bufs[_index_bias(1)]);
    }
  } else {
    if (n < 5 && px->has_bias) return -1; // A,B,D,bias,params → 최소 5
    if (n < 4 && !px->has_bias) return -1; // A,B,D,params → 최소 4
    D = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[_index_D(0)])));
    if (px->has_bias) {
      bias = reinterpret_cast<const float*>(bufs[_index_bias(0)]);
    }
  }

  // 파라미터 구성
  GemmBiasActParams p{};
  p.M = px->M; p.N = px->N; p.K = px->K;

  // stride: 0/음수면 row-major 기본값
  const int lda = (px->lda > 0 ? px->lda : px->K);
  const int ldb = (px->ldb > 0 ? px->ldb : px->N);
  const int ldc = (px->ldc > 0 ? px->ldc : px->N);
  const int ldd = (px->ldd > 0 ? px->ldd : px->N);

  p.A = const_cast<float*>(A); p.lda = lda;
  p.B = const_cast<float*>(B); p.ldb = ldb;
  p.C = const_cast<float*>(C); p.ldc = ldc;
  p.D = D;                     p.ldd = ldd;

  p.alpha = (px->alpha == 0.f ? 1.f : px->alpha);
  p.beta  = px->beta;

  p.bias = const_cast<float*>(bias);
  p.bias_kind = _map_bias(px->bias_kind);
  p.act  = _map_act(px->act_kind);

  // 현재 regemm LeakyReLU slope 커스터마이즈가 없으면 px->leaky_slope는 무시됨

  // 통합 엔트리 호출
  gemm_bias_act_f32(p, s);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}
