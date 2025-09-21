// backends/cuda/ops/gemm/backward.cu
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp"      // StreamHandle, ai::Status

#include "regemm/api.h"         // regemm::{GemmBiasActBwdParams, gemm_bias_act_bwd_f32, ActKind, BiasKind}

namespace {

// RowMajor 2D 텐서의 leading dim 추론
inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1]; // contiguous row-major 가정
}

// ai::ActKind -> regemm::ActKind
inline regemm::ActKind to_regemm_act(ai::ActKind a) {
  using A = ai::ActKind; using R = regemm::ActKind;
  switch (a) {
    case A::None:      return R::None;
    case A::ReLU:      return R::ReLU;
    case A::LeakyReLU: return R::LeakyReLU;
    case A::GELU:      return R::GELU;
    case A::Sigmoid:   return R::Sigmoid;
    case A::Tanh:      return R::Tanh;
  }
  return R::None;
}

// (권장) FWD에서 실제로 사용한 bias 텐서 모양을 기준으로 판정
inline regemm::BiasKind deduce_bias_kind_from_forward(const ai::Tensor* bias_like,
                                                      int64_t M, int64_t N) {
  if (!bias_like || !bias_like->data) return regemm::BiasKind::None;
  if (bias_like->desc.shape.size() == 1) {
    const auto sz = bias_like->desc.shape[0];
    if (sz == 1) return regemm::BiasKind::Scalar; // 최우선
    if (sz == N) return regemm::BiasKind::PerN;   // M==N이어도 PerN 우선
    if (sz == M) return regemm::BiasKind::PerM;
  }
  return regemm::BiasKind::None;
}

} // anonymous

namespace ai {

/**
 * GEMM-bias-activation Backward (f32, row-major, 비전치 전제)
 *
 * 입력:
 *   A:[M,K], B:[K,N], (C:[M,N]|optional), gY:[M,N], Z:[M,N]
 * 출력:
 *   gA:[M,K]|optional, gB:[K,N]|optional, gC:[M,N]|optional, gBias:[1|M|N]|optional
 *
 * 제약/메모:
 *   - 현재 f32 / RowMajor / trans_a,b=false 만 지원
 *   - forward에서 C/beta를 쓰지 않았다면 gC는 계산해도 0이므로 생략 권장
 *   - bias 축은 FWD와 동일해야 함(Scalar / PerM / PerN). 가능하면 FWD의 bias 텐서를 함께 전달하는 게 가장 정확.
 *
 *   Note: 시그니처는 기존 GemmAttrs를 재사용하지만, FWD의 alpha/beta를 내려받아야 정확히 일치합니다.
 *         지금은 p.alpha=1, p.beta=(C&&gC?1:0)로 두었으니, 필요 시 전용 GemmBwdAttrs(alpha,beta 포함)로 교체하세요.
 */
ai::Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                            const Tensor& gY, const Tensor& Z,
                            Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                            const GemmAttrs& attrs, StreamHandle stream)
{
  // 1) 디바이스/타입/레이아웃 가드
  auto is_cuda_f32_rm = [](const Tensor& T){
    return T.is_cuda() && T.desc.dtype==DType::F32 && T.desc.layout==Layout::RowMajor;
  };
  if (!is_cuda_f32_rm(A) || !is_cuda_f32_rm(B) || !is_cuda_f32_rm(gY) || !is_cuda_f32_rm(Z))
    return ai::Status::DeviceMismatch;
  if (gA && !is_cuda_f32_rm(*gA)) return ai::Status::DeviceMismatch;
  if (gB && !is_cuda_f32_rm(*gB)) return ai::Status::DeviceMismatch;
  if (gC && !is_cuda_f32_rm(*gC)) return ai::Status::DeviceMismatch;
  if (C  && !is_cuda_f32_rm(*C))  return ai::Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b) return ai::Status::TransposeNotSupported; // 현재 비전치 전제

  // 2) 치수 체크
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 ||
      gY.desc.shape.size()!=2 || Z.desc.shape.size()!=2) return ai::Status::ShapeMismatch;

  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
  if (K != Kb) return ai::Status::ShapeMismatch;

  if (gY.desc.shape[0]!=M || gY.desc.shape[1]!=N) return ai::Status::ShapeMismatch;
  if (Z .desc.shape[0]!=M || Z .desc.shape[1]!=N) return ai::Status::ShapeMismatch;

  if (gA && (gA->desc.shape.size()!=2 || gA->desc.shape[0]!=M || gA->desc.shape[1]!=K)) return ai::Status::ShapeMismatch;
  if (gB && (gB->desc.shape.size()!=2 || gB->desc.shape[0]!=K || gB->desc.shape[1]!=N)) return ai::Status::ShapeMismatch;
  if (gC) {
    if (!C) return ai::Status::MissingInput; // gC를 원하면 C도 있어야 함(커널이 C&&gC 둘 다 있을 때만 gC 씀)
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

  // 4) bias kind 추론
  //   (A) 가장 정확: FWD에 사용한 bias 텐서를 받아서 판정 → deduce_bias_kind_from_forward(...)
  //   (B) FWD bias 텐서를 알 수 없을 때: gBias shape[1|M|N]만으로 보수적으로 추론 (fallback)
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (gBias && gBias->data) {
    if (gBias->desc.shape.size()==1) {
      const auto sz = gBias->desc.shape[0];
      if      (sz == 1) bk = regemm::BiasKind::Scalar;
      else if (sz == N) bk = regemm::BiasKind::PerN;   // PerN 우선
      else if (sz == M) bk = regemm::BiasKind::PerM;
    }
  }
  // TODO: 가능하면 여기서 FWD bias 텐서를 함께 인자로 받아
  //   bk = deduce_bias_kind_from_forward(fwd_bias, M, N);
  // 로 대체하세요.

  // 5) 파라미터 구성
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

  // 6) 에필로그/스케일 파라미터
  //  - FWD와 alpha/beta가 동일해야 정확. 지금은 기본값/가정값.
  p.alpha = 1.0f;                               // FWD alpha를 정확히 전달할 수 있으면 그 값 사용
  p.beta  = (C && gC) ? 1.0f : 0.0f;            // FWD에서 beta를 썼다면 동일 값 전달 필요

  p.bias_kind   = bk;
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // 7) 실행
  regemm::gemm_bias_act_bwd_f32(p, reinterpret_cast<cudaStream_t>(stream));
  return ai::Status::Ok;
}

} // namespace ai
