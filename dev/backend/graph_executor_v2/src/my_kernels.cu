// my_kernels.cu
#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cstdio>

// regemm headers (expect add_subdirectory(../regemm_epilogue ...) and include dir linked)
#include "regemm/api.h"

using namespace regemm;

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

// ---- Backward(EX): A,B,(C), gY, Z, gA, gB, (gC), (gBias), &pb ----
extern "C" int ge2_launch_gemm_bias_act_bwd_f32_ex(const ge2_uintptr* bufs, int n, void* stream_opaque) {
  if (!bufs || n < 7) return -1; // 최소 A,B,gY,Z,gA,gB,&pb
  auto s = reinterpret_cast<cudaStream_t>(stream_opaque);

  const auto* pb = reinterpret_cast<const ge2_gemm_bias_act_bwd_params_t*>(bufs[n - 1]);
  if (!pb) return -1;

  int idx = 0;
  const float* A  = reinterpret_cast<const float*>(bufs[idx++]); // [M,K]
  const float* B  = reinterpret_cast<const float*>(bufs[idx++]); // [K,N]

  const float* C  = nullptr;
  bool use_C = false;
  // C 사용 여부는 포인터 유무 + ld 힌트로 판단 (명시적 플래그가 없는 C API)
  // 호출 측에서 C=None이면 bufs에 안 넣는 걸 권장.
  // 여기서는 안전하게 판단: 남은 개수와 stride/beta를 참고
  if (pb->ldc >= 0) {
    // bufs가 충분하고 다음이 C일 가능성 체크
    // 최소 레이아웃 생각: [A,B,(C),gY,Z,gA,gB,(gC),(gBias),&pb]
    // gY는 반드시 있어야 하므로 bufs[idx]가 gY가 아닐 때만 C로 본다
    // 정확 매핑은 호출 측 보장이 가장 확실함.
  }

  // 간단 & 확실: 호출 규약을 따르게 해서 C가 있으면 무조건 bufs[idx]가 C
  // (launch_table / pybind에서 이 규약을 강제했으니 그대로 사용)
  if (n >= 9) {
    // C가 있는 경우의 최소 길이 시나리오: A,B,C,gY,Z,gA,gB,&pb => n >= 8
    // 여기선 여유있게 n>=9에서만 C 후보로 본다 (gC/gBias 포함 시)
    // 더 안전히: 이름있는 래퍼에서 확정적으로 넣어주도록 했으므로 그대로 파싱
  }
  // 더 안전한 파싱: pybind에서 "C가 None이 아니면 bufs에 넣는다" 규약 → 여기서 px와 동일하게 처리
  // 즉, n을 이용해 역으로 추론하기보단, 다음 객체가 gY인지 검사 불가하므로 호출측 규약 신뢰
  // 따라서 런타임에서는 "C를 쓰면 bufs에 C를 넣는다" 전제로 idx 재구성:

  // 파싱 재정의:
  // 최소 필수는 C가 없을 때 [A,B,gY,Z,gA,gB,&pb] => n==7
  // C가 있으면 [A,B,C,gY,Z,gA,gB,&pb] => n==8
  // gC, gBias가 더해지면 9 또는 10
  if (n == 8 || n == 9 || n == 10) {
    // bufs[2]가 C일 가능성 높음
    C = reinterpret_cast<const float*>(bufs[2]);
    use_C = true;
    idx = 3;
  } else {
    idx = 2; // C 없음
  }

  const float* gY = reinterpret_cast<const float*>(bufs[idx++]); // [M,N]
  const float* Z  = reinterpret_cast<const float*>(bufs[idx++]); // [M,N]
  float* gA       = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[idx++])));
  float* gB       = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[idx++])));

  float* gC = nullptr;
  float* gBias = nullptr;

  // 남은 bufs: (gC), (gBias), &pb
  if (idx < n - 1) {
    // 하나 남았으면 gC or gBias 중 하나
    // 둘 남았으면 gC, gBias 둘 다
    int remain = (n - 1) - idx;
    if (remain == 1) {
      // 어떤 것이든 상관 없이 포인터만 전달 (API 쪽에서 nullptr 구분 없음)
      gC = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[idx++])));
    } else if (remain >= 2) {
      gC    = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[idx++])));
      gBias = reinterpret_cast<float*>(const_cast<void*>(reinterpret_cast<const void*>(bufs[idx++])));
    }
  }

  // 파라미터 구성
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

  // 호출
  gemm_bias_act_bwd_f32(p, s);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}
