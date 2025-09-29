// src/ops/gemm.cpp
#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp" 

namespace ai { namespace ops {

// 간단한 유효성 검사만 하고 Registry로 디스패치
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle stream)
{
  // (0) 널 포인터 방어
  if (!A.data || !B.data || !Y.data) return -1;         // 빈 텐서 방지
  // (0.1) 아직 미지원: 전치 경로
  if (attrs.trans_a || attrs.trans_b) return -11;

  // (1) 타입/레이아웃/디바이스 검증 (현재 f32/row-major/cuda만 연결)
  if (A.desc.dtype != DType::F32 || B.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return -2;
  if (A.desc.layout != Layout::RowMajor || B.desc.layout != Layout::RowMajor || Y.desc.layout != Layout::RowMajor) return -3;
  if (A.device != Device::CUDA || B.device != Device::CUDA || Y.device != Device::CUDA) return -4;

  // (2) shape 검증
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2) return -5;
  const int64_t M = A.desc.shape[0], K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0], N = B.desc.shape[1];
  if (K != Kb) return -6;
  if (Y.desc.shape[0] != M || Y.desc.shape[1] != N) return -7;

  // (3) Bias는 1D (1|M|N) 혹은 None만 허용 (상위 바인딩도 체크하지만 안전하게)
  const Tensor* effBias = nullptr; // 비어있으면 nullptr로 정규화
  if (Bias && Bias->data) {
    if (Bias->desc.dtype != DType::F32) return -8;
    if (Bias->desc.shape.size() != 1) return -9;
    const auto bl = Bias->desc.shape[0];
    if (!(bl == 1 || bl == M || bl == N)) return -10;
    effBias = Bias;
  }

  // (4) 디스패치
  OpQuery q{OpKind::GEMM, A, B, effBias, Y, attrs};
  ai::KernelFn fn = OpRegistry::inst().find_best(q);
  if (!fn) return -100;  // 등록 누락/미지원 조합

  ai::Status st = fn(A, B, effBias, Y, attrs, stream);
  return (st == ai::Status::Ok) ? 0 : -7; // 필요 시 상세 매핑
}

}} // namespace ai::ops
