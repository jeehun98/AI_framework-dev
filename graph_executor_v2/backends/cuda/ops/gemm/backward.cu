// backends/cuda/ops/gemm/backward.cu
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/gemm/gemm_common.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

#include "regemm/api.h"

// ★ 추가: 공용 유틸


namespace {

using namespace ai::gemm_common;

} // anonymous

namespace ai {

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

  // 4) bias kind (가능하면 FWD bias 텐서를 받아 deduce 사용 권장)
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (gBias && gBias->data) {
    bk = infer_bias_kind_1d_lenMN(gBias, M, N);  // fallback
  }
  // TODO: FWD bias 텐서를 받는다면 아래로 교체:
  // bk = deduce_bias_kind_from_forward(fwd_bias, M, N);

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

  // 6) 에필로그/스케일 (FWD와 동일 값이 가장 정확)
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
