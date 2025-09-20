// backends/cuda/ops/gemm/launcher.cu
//
// 역할:
//  - 상위 ai::Tensor/GemmAttrs를 regemm::GemmBiasActParamsEx로 매핑
//  - (현재) f32, row-major, 비전치 경로 지원
//  - Bias 1D 축 판정: Scalar(1) > PerN(len==N) > PerM(len==M)
//    * M==N인 경우 PerN을 기본으로 우선
//    * N==1 또는 M==1일 때도 Scalar가 먼저 잡히도록 순서가 중요!
//
// 반환코드 약속(0=OK, <0 오류):
//  -1 : 디바이스가 CUDA가 아님
//  -2 : dtype(f32) 불일치
//  -3 : 레이아웃(row-major) 불일치
//  -4 : transpose 경로 미지원
//  -5 : shape 차원 불일치(2D 아님 등)
//  -6 : 행렬 크기 불일치(M,K,N 검사 실패)
//  -7 : leading dim(ld*) 유효성 실패
//  -8 : 정수 변환 범위 초과(int32)
//
// 주의:
//  - stream은 상위에서 void*로 전달되며 여기서 cudaStream_t로 재해석.
//  - regemm EX 파라미터는 Z-stash도 지원하지만, 현재 save_preact=0으로 비활성.
//

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <limits>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"


#include "regemm/api.h"

namespace {

// --- 유틸: row-major 2D 텐서의 leading dimension 추론 ---
// 우선 stride[0]이 있으면 그 값을 사용, 없으면 기본 contiguous로 shape[1].
inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1];
}

// --- ai::ActKind → regemm::ActKind 매핑 ---
inline ge2::regemm::ActKind to_regemm_act(ai::ActKind a) {
  using A = ai::ActKind;
  using R = ge2::regemm::ActKind;
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

// --- Bias 축 판정 규칙 ---
//  * 길이 1 ⇒ Scalar (항상 최우선; N==1/M==1 케이스 보호)
//  * 길이==N ⇒ PerN (M==N 동률인 경우에도 PerN 우선)
//  * 길이==M ⇒ PerM
//  * 그 외/2D 이상 ⇒ None (보수적 무시)
inline ge2::regemm::BiasKind infer_bias_kind(const ai::Tensor* Bias, int64_t M, int64_t N) {
  if (!Bias || !Bias->data) return ge2::regemm::BiasKind::None;
  const auto& d = Bias->desc;
  if (d.shape.size() != 1) return ge2::regemm::BiasKind::None;

  const int64_t len = d.shape[0];
  if (len == 1) return ge2::regemm::BiasKind::Scalar; // ★ scalar 먼저!
  if (len == N) return ge2::regemm::BiasKind::PerN;   // ★ M==N이면 PerN 우선
  if (len == M) return ge2::regemm::BiasKind::PerM;
  return ge2::regemm::BiasKind::None;
}

// --- int64→int32 안전 변환 체크 ---
// regemm 파라미터는 int32 필드이므로 범위를 초과하면 에러로 처리.
inline bool fits_int32(int64_t x) {
  return x >= std::numeric_limits<int>::min() && x <= std::numeric_limits<int>::max();
}

} // anonymous namespace

namespace ai {

// 0=OK, <0 error
Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                      Tensor& Y, const GemmAttrs& attrs, StreamHandle stream) {
  // 1) 기본 가드: 디바이스/타입/레이아웃/transpose 지원여부
  if (!A.is_cuda() || !B.is_cuda() || !Y.is_cuda()) return -1;
  if (A.desc.dtype != DType::F32 || B.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return -2;
  if (A.desc.layout != Layout::RowMajor || B.desc.layout != Layout::RowMajor || Y.desc.layout != Layout::RowMajor) return -3;
  if (attrs.trans_a || attrs.trans_b) return -4; // 현재 비전치만 지원

  // 2) shape 검증
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2) return -5;
  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
  if (K!=Kb || Y.desc.shape[0]!=M || Y.desc.shape[1]!=N) return -6;

  // 3) leading dim 추론 및 유효성 체크
  const int64_t lda = infer_ld_rowmajor_2d(A);
  const int64_t ldb = infer_ld_rowmajor_2d(B);
  const int64_t ldd = infer_ld_rowmajor_2d(Y);
  if (lda < K || ldb < N || ldd < N) return -7;

  // 4) regemm 파라미터의 int32 제한 확인
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldd)) {
    return -8;
  }

  // 5) regemm 확장 파라미터 구성
  ge2::regemm::GemmBiasActParamsEx p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  // A, B, (C 미사용), D
  p.A   = A.data; p.lda = static_cast<int>(lda);
  p.B   = B.data; p.ldb = static_cast<int>(ldb);
  p.C   = nullptr; p.ldc = 0;          // C는 현재 미사용 (beta=0)
  p.D   = Y.data; p.ldd = static_cast<int>(ldd);

  // 스케일 (상위에서 alpha/beta 노출 X → alpha=1, beta=0)
  p.alpha = 1.0f;
  p.beta  = 0.0f;

  // Bias 포인터 + 축 판정
  p.bias      = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind = infer_bias_kind(Bias, M, N); // ★ 규칙 반영 (Scalar > PerN > PerM)

  // Activation
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // Z stash (EX 기능) — 현재 비활성. 오토그래드 연동 시 여기 활성화.
  p.Z           = nullptr;
  p.ldZ         = 0;    // 0이면 내부에서 ldd로 간주
  p.save_preact = 0;    // 1이면 pre-activation(Z) 저장

  // 6) 실행 — stream은 void* → cudaStream_t 재해석
  ge2::regemm::gemm_bias_act_f32_ex(p, reinterpret_cast<cudaStream_t>(stream));
  return 0;
}

} // namespace ai
