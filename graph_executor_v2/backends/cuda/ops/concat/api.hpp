#pragma once

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

// ============================================================================
// Concat (CUDA, row-major contiguous)
// ----------------------------------------------------------------------------
// - Forward : Y = concat(Xs, axis)
// - Backward: gX_i += slice(gY, axis, offset_i, size_i)
// - Layout  : 모든 텐서는 RowMajor 연속, 동일 rank
// - DType   : 현재 F32 고정 (혼합정밀/타입 확장은 이후 단계에서)
// - Capture : 동적 할당 없음, 고정 shape/attrs 가정
// - Notes   : 음수 axis 허용(-rank..rank-1), 내부에서 정규화
// ============================================================================

// I64 미지원: attrs는 int32
struct ConcatAttrs {
  int rank{1};   // Xs[i], Y 모두 동일 rank
  int axis{0};   // 유효 범위: -rank .. rank-1 (내부 정규화)
};

// ------------------------------- Contracts ----------------------------------
// Inputs:
//   - Xs: 길이 n (>0) 의 텐서 배열
//   - 모두 Device::CUDA, DType::F32, Layout::RowMajor, 연속
//   - 모든 Xs의 rank 동일 == attrs.rank
//   - 축(axis) 제외 모든 차원 동일
// Output:
//   - Y.shape[axis] = Σ_i Xs[i].shape[axis]; 나머지는 입력과 동일
// Aliasing:
//   - Y는 어떤 Xs[i]와도 동일 버퍼 금지
// Backward:
//   - gXs[i]가 nullptr가 아니면 해당 슬라이스에 누적 += 수행(Zero-init은 호출자 책임)
//   - gY, gXs 모두 F32, RowMajor, 연속

// --------------------------------- API --------------------------------------

// Forward: Y = concat(Xs, axis)
Status ConcatCudaLaunch(const Tensor* Xs, int n,
                        Tensor& Y,
                        const ConcatAttrs& attrs,
                        StreamHandle stream);

// Backward: 각 gX_i += slice(gY, axis, offset_i, size_i)
//  - gXs 길이 n, nullptr 허용(스킵)
Status ConcatCudaBackwardLaunch(const Tensor& gY,
                                Tensor* gXs, int n,
                                const ConcatAttrs& attrs,
                                StreamHandle stream);

// ---------------------------- (선택) 유틸리티 --------------------------------
// 출력 shape 계산만 필요할 때 사용 가능(런타임 검증은 런처 내부에서도 수행됨).
// 실패 시 ShapeMismatch/DtypeMismatch/LayoutMismatch 등 반환.
inline Status ConcatInferOutputShape(const Tensor* Xs, int n,
                                     const ConcatAttrs& attrs,
                                     /*out*/ std::vector<int64_t>& out_shape)
{
  if (!Xs || n <= 0) return Status::MissingInput;
  const Tensor& X0 = Xs[0];
  if (!X0.is_cuda()) return Status::DeviceMismatch;
  if (X0.desc.layout != Layout::RowMajor) return Status::LayoutMismatch;
  if (X0.desc.dtype  != DType::F32) return Status::DtypeMismatch;

  const int rk = attrs.rank;
  if ((int)X0.desc.shape.size() != rk) return Status::ShapeMismatch;

  int axis = attrs.axis;
  if (axis < 0) axis += rk;
  if (axis < 0 || axis >= rk) return Status::InvalidArgument;

  out_shape = X0.desc.shape;
  int64_t axis_sum = 0;

  for (int i = 0; i < n; ++i) {
    const Tensor& Xi = Xs[i];
    if (!Xi.is_cuda()) return Status::DeviceMismatch;
    if (Xi.desc.layout != Layout::RowMajor) return Status::LayoutMismatch;
    if (Xi.desc.dtype  != DType::F32) return Status::DtypeMismatch;
    if ((int)Xi.desc.shape.size() != rk) return Status::ShapeMismatch;

    for (int d = 0; d < rk; ++d) {
      if (d == axis) continue;
      if (Xi.desc.shape[d] != X0.desc.shape[d]) return Status::ShapeMismatch;
    }
    axis_sum += Xi.desc.shape[axis];
  }

  out_shape[axis] = axis_sum;
  return Status::Ok;
}

} // namespace ai
