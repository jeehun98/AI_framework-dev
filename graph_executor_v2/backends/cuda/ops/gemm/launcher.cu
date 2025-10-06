// backends/cuda/ops/gemm/launcher.cu  (FWD+BWD 통합)
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <limits>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/gemm/gemm_common.hpp"
#include "backends/cuda/ops/gemm/api.hpp"
#include "regemm/api.h"

namespace {
using namespace ai::gemm_common;

// 기존 infer_bias_kind_1d_lenMN가 너무 보수적일 수 있어
// (N,), (1,N), (M,), (M,1), (1,1) 모두를 관대히 매칭하는 fallback 버전 추가
inline regemm::BiasKind infer_bias_kind_fallback(const ai::Tensor* Bias, int64_t M, int64_t N) {
  using BK = regemm::BiasKind;
  if (!Bias || !Bias->data) return BK::None;

  const auto& s = Bias->desc.shape;  // 허용: 1D 또는 2D
  int64_t numel = 1;
  for (auto v : s) numel *= v;
  if (numel <= 0) return BK::None;

  // 우선 정확 매칭
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

} // anonymous

namespace ai {

// =========================
// Forward (save_z 지원)
// =========================
ai::Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                          Tensor& Y, const GemmAttrs& attrs,
                          StreamHandle stream, Tensor* Z_saved /*=nullptr*/) {
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
  // NOTE: Z_saved may alias Y. EX 커널은 Z(pre)를 먼저 쓰고, 다음에 D=act(pre)를 계산하므로 안전.
  if (attrs.save_z && Z_saved == nullptr) {
    return ai::Status::MissingOutput; // 명시적으로 에러 리턴
  }
  const bool want_save_z = attrs.save_z && (Z_saved != nullptr);

  int   ldZ_i = 0;
  void* Z_ptr = nullptr;
  if (want_save_z) {
    if (!is_cuda_f32_rowmajor(*Z_saved)) return ai::Status::DeviceMismatch;
    if (Z_saved->desc.shape.size()!=2 ||
        Z_saved->desc.shape[0]!=M || Z_saved->desc.shape[1]!=N) {
      return ai::Status::ShapeMismatch;
    }
    const int64_t ldZ = infer_ld_rowmajor_2d(*Z_saved);
    if (ldZ < N) return ai::Status::StrideMismatch;
    if (!fits_int32(ldZ)) return ai::Status::Invalid;
    ldZ_i = static_cast<int>(ldZ);
    Z_ptr = Z_saved->data;
  }

  // 6) regemm 파라미터
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

  // ---- bias 전달 + kind 추론(관대) ----
  p.bias      = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind = infer_bias_kind_fallback(Bias, M, N);

  // 일부 커널은 명시 플래그를 추가로 요구할 수 있음 (필드가 있을 때만 세팅)
  // CMake 등에서 -DREGE_MM_PARAMS_HAS_WITH_BIAS 정의 시 활성화
  #ifdef REGE_MM_PARAMS_HAS_WITH_BIAS
  p.with_bias = (p.bias != nullptr && p.bias_kind != regemm::BiasKind::None) ? 1 : 0;
  #endif

  // ---- activation / leaky slope ----
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // ---- Z 저장: pre-activation을 단일 패스로 저장 ----
  p.Z           = want_save_z ? Z_ptr : nullptr;
  p.ldZ         = want_save_z ? ldZ_i : 0;
  p.save_preact = want_save_z ? 1      : 0;

  // 7) 실행
  regemm::gemm_bias_act_f32_ex(p, reinterpret_cast<cudaStream_t>(stream));
  return ai::Status::Ok;
}

// =========================
// Backward
// =========================
ai::Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                            const Tensor& gY, const Tensor& Z,
                            Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                            const GemmAttrs& attrs, StreamHandle stream)
{
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

  // (추가) int32 범위 가드
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldgY) || !fits_int32(ldZ) ||
      (gA && !fits_int32(ldgA)) || (gB && !fits_int32(ldgB)) || (gC && !fits_int32(ldgC))) {
    return ai::Status::Invalid;
  }

  // 4) bias kind (gBias가 있을 때만 의미 있음)
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (gBias && gBias->data) {
    // gBias shape 기반 PerN/PerM/Scalar 판정
    bk = infer_bias_kind_fallback(gBias, M, N);
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

  // 7) 실행
  regemm::gemm_bias_act_bwd_f32(p, reinterpret_cast<cudaStream_t>(stream));
  return ai::Status::Ok;
}

} // namespace ai
