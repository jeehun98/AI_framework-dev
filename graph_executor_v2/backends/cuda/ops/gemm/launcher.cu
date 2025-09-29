// backends/cuda/ops/gemm/launcher.cu
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

// 기존: infer_ld_rowmajor_2d, to_regemm_act, infer_bias_kind, fits_int32
// → 위 공용 유틸 사용으로 대체됨

inline regemm::BiasKind infer_bias_kind(const ai::Tensor* Bias, int64_t M, int64_t N) {
  return infer_bias_kind_1d_lenMN(Bias, M, N);
}

} // anonymous

namespace ai {

ai::Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                          Tensor& Y, const GemmAttrs& attrs, StreamHandle stream) {
  // 1) 기본 가드
  if (!is_cuda_f32_rowmajor(A) || !is_cuda_f32_rowmajor(B) || !is_cuda_f32_rowmajor(Y))
    return ai::Status::DeviceMismatch;
  if (attrs.trans_a || attrs.trans_b) return ai::Status::TransposeNotSupported;

  // 2) shape
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2)
    return ai::Status::ShapeMismatch;
  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
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

  // 5) regemm 파라미터
  regemm::GemmBiasActParamsEx p{};
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
  p.bias_kind = infer_bias_kind(Bias, M, N);

  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  p.Z           = nullptr;
  p.ldZ         = 0;
  p.save_preact = 0;

  regemm::gemm_bias_act_f32_ex(p, reinterpret_cast<cudaStream_t>(stream));
  return ai::Status::Ok;
}

} // namespace ai
