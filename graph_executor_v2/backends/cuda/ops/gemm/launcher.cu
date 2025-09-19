// backends/cuda/ops/gemm/launcher.cu
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

#include "regemm/api.h"  // GemmBiasActParamsEx / gemm_bias_act_f32_ex

namespace {

inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1];
}

inline regemm::ActKind to_regemm_act(ai::ActKind a) {
  using A = ai::ActKind;
  using R = regemm::ActKind;
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

inline regemm::BiasKind deduce_bias_kind(const ai::Tensor* Bias, int64_t M, int64_t N) {
  if (!Bias || !Bias->data) return regemm::BiasKind::None;
  if (Bias->desc.shape.size() == 1) {
    const auto sz = Bias->desc.shape[0];
    if (sz == N) return regemm::BiasKind::PerN;
    if (sz == M) return regemm::BiasKind::PerM;
    if (sz == 1) return regemm::BiasKind::Scalar;
  }
  // 형태가 애매하면 보수적으로 None
  return regemm::BiasKind::None;
}

} // anonymous

namespace ai {

// 0=OK, <0 error
Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                      Tensor& Y, const GemmAttrs& attrs, StreamHandle stream) {
  // 1) 가드
  if (!A.is_cuda() || !B.is_cuda() || !Y.is_cuda()) return -1;
  if (A.desc.dtype != DType::F32 || B.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return -2;
  if (A.desc.layout != Layout::RowMajor || B.desc.layout != Layout::RowMajor || Y.desc.layout != Layout::RowMajor) return -3;
  if (attrs.trans_a || attrs.trans_b) return -4; // 현재 비전치 경로만

  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2) return -5;
  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
  if (K!=Kb || Y.desc.shape[0]!=M || Y.desc.shape[1]!=N) return -6;

  const int64_t lda = infer_ld_rowmajor_2d(A);
  const int64_t ldb = infer_ld_rowmajor_2d(B);
  const int64_t ldd = infer_ld_rowmajor_2d(Y);
  if (lda < K || ldb < N || ldd < N) return -7;

  // 2) regemm 확장 파라미터 구성 (Z stash off)
  regemm::GemmBiasActParamsEx p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  p.A = A.data; p.lda = static_cast<int>(lda);
  p.B = B.data; p.ldb = static_cast<int>(ldb);
  p.C = nullptr; p.ldc = 0; // C 미사용
  p.D = Y.data; p.ldd = static_cast<int>(ldd);

  p.alpha = 1.0f; // 상위 scale 미지원 → 1
  p.beta  = 0.0f; // C 미사용 → 0

  p.bias       = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind  = deduce_bias_kind(Bias, M, N);

  p.act        = to_regemm_act(attrs.act);
  p.leaky_slope= attrs.leaky_slope;

  p.Z          = nullptr; // 요청 시 후속 확장
  p.ldZ        = 0;
  p.save_preact= 0;

  // 3) 실행
  regemm::gemm_bias_act_f32_ex(p, reinterpret_cast<cudaStream_t>(stream));
  return 0;
}

} // namespace ai
