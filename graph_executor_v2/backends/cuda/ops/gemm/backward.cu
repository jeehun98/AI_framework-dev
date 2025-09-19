// backends/cuda/ops/gemm/backward.cu
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp"      // StreamHandle, Status

#include "regemm/api.h"         // GemmBiasActBwdParams / gemm_bias_act_bwd_f32

namespace {

inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1]; // contiguous row-major 가정
}

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

inline regemm::BiasKind deduce_bias_kind_from_forward(const ai::Tensor* bias_like, int64_t M, int64_t N) {
  // fwd에서 사용했던 bias 형태를 그대로 전달해야 gBias 축적 크기가 맞음
  if (!bias_like || !bias_like->data) return regemm::BiasKind::None;
  if (bias_like->desc.shape.size() == 1) {
    const auto sz = bias_like->desc.shape[0];
    if (sz == N) return regemm::BiasKind::PerN;
    if (sz == M) return regemm::BiasKind::PerM;
    if (sz == 1) return regemm::BiasKind::Scalar;
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
 * 제약:
 *   - 현재 f32 / RowMajor / trans_a,b=false 만 지원
 *   - forward에서 C/beta를 쓰지 않았다면 gC는 계산해도 0이므로 생략 권장
 */
Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                        const Tensor& gY, const Tensor& Z,
                        Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                        const GemmAttrs& attrs, StreamHandle stream)
{
  // 1) 디바이스/타입/레이아웃 가드
  auto is_cuda_f32_rm = [](const Tensor& T){
    return T.is_cuda() && T.desc.dtype==DType::F32 && T.desc.layout==Layout::RowMajor;
  };
  if (!is_cuda_f32_rm(A) || !is_cuda_f32_rm(B) || !is_cuda_f32_rm(gY) || !is_cuda_f32_rm(Z))
    return -101;
  if (gA && !is_cuda_f32_rm(*gA)) return -102;
  if (gB && !is_cuda_f32_rm(*gB)) return -103;
  if (gC && !is_cuda_f32_rm(*gC)) return -104;
  if (C  && !is_cuda_f32_rm(*C))  return -105;
  if (attrs.trans_a || attrs.trans_b) return -106; // 현재 비전치 전제

  // 2) 치수 체크
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 ||
      gY.desc.shape.size()!=2 || Z.desc.shape.size()!=2) return -107;

  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
  if (K != Kb) return -108;

  if (gY.desc.shape[0]!=M || gY.desc.shape[1]!=N) return -109;
  if (Z .desc.shape[0]!=M || Z .desc.shape[1]!=N) return -110;

  if (gA && (gA->desc.shape.size()!=2 || gA->desc.shape[0]!=M || gA->desc.shape[1]!=K)) return -111;
  if (gB && (gB->desc.shape.size()!=2 || gB->desc.shape[0]!=K || gB->desc.shape[1]!=N)) return -112;
  if (gC) {
    if (!C) return -113; // gC를 원하면 C도 있어야 함(커널이 C&&gC 둘 다 있을 때만 gC 씀)
    if (gC->desc.shape.size()!=2 || gC->desc.shape[0]!=M || gC->desc.shape[1]!=N) return -114;
  }

  // 3) leading dims
  const int64_t lda  = infer_ld_rowmajor_2d(A);
  const int64_t ldb  = infer_ld_rowmajor_2d(B);
  const int64_t ldgY = infer_ld_rowmajor_2d(gY);
  const int64_t ldZ  = infer_ld_rowmajor_2d(Z);
  if (lda < K || ldb < N || ldgY < N || ldZ < N) return -115;

  int64_t ldgA = 0, ldgB = 0, ldgC = 0;
  if (gA) { ldgA = infer_ld_rowmajor_2d(*gA); if (ldgA < K) return -116; }
  if (gB) { ldgB = infer_ld_rowmajor_2d(*gB); if (ldgB < N) return -117; }
  if (gC) { ldgC = infer_ld_rowmajor_2d(*gC); if (ldgC < N) return -118; }

  // 4) bias kind 추론 (fwd에 사용했던 bias 텐서를 알고 있으면 그 모양을 넘겨주세요)
  //    여기서는 gBias 텐서 모양만으로도 추론 시도
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (gBias && gBias->data) {
    // gBias shape이 [1]/[M]/[N] 중 하나라고 가정
    if (gBias->desc.shape.size()==1) {
      const auto sz = gBias->desc.shape[0];
      if      (sz == 1) bk = regemm::BiasKind::Scalar;
      else if (sz == M) bk = regemm::BiasKind::PerM;
      else if (sz == N) bk = regemm::BiasKind::PerN;
    }
    // 모양이 불일치하면 None으로 두고 커널에서 아무것도 안 누적
  }

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

  // forward의 scale 파라미터와 일치해야 하나,
  // 현재 상위 스키마에 노출되어 있지 않으므로 기본값으로 둡니다.
  p.alpha = 1.0f;
  p.beta  = (C && gC) ? 1.0f : 0.0f; // fwd에서 beta를 사용하지 않았다면 0.0으로 두세요.

  p.bias_kind   = bk;
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // 6) 실행
  regemm::gemm_bias_act_bwd_f32(p, reinterpret_cast<cudaStream_t>(stream));
  return 0;
}

} // namespace ai
