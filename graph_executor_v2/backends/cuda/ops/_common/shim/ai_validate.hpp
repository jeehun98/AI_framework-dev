#pragma once
#include "ai_status.hpp"
#include "ai_tensor.hpp"

namespace ai {

// ---------------- Common validators ----------------
inline Status expect_rowmajor_2d(const Tensor& T, const char* /*name*/) {
  if (!T.is_cuda()) return Status::DeviceMismatch;
  if (!T.is_contiguous_rowmajor_2d()) return Status::LayoutMismatch;
  if (T.desc.dtype != DType::F32 && T.desc.dtype != DType::F16 && T.desc.dtype != DType::BF16)
    return Status::DtypeMismatch;
  if (T.numel() <= 0) return Status::ShapeMismatch;
  return Status::Ok;
}

// Bias는 Per-OUT((1,Out) or (Out,))만 허용
inline Status expect_bias_per_out_or_null(const Tensor* B, int64_t Out) {
  if (!B) return Status::Ok;
  const Tensor& t = *B;
  if (!t.is_cuda()) return Status::DeviceMismatch;
  if (t.desc.shape.size()==1) {
    if (t.desc.shape[0] != Out) return Status::ShapeMismatch;
  } else if (t.desc.shape.size()==2) {
    if (!(t.desc.shape[0]==1 && t.desc.shape[1]==Out)) return Status::ShapeMismatch;
  } else return Status::ShapeMismatch;
  return Status::Ok;
}

// 3D row-major, dtype ∈ {F32,F16,BF16}
inline Status expect_rowmajor_3d_f32_any(const Tensor& T) {
  if (!T.is_cuda()) return Status::DeviceMismatch;
  if (T.desc.layout != Layout::RowMajor) return Status::LayoutMismatch;
  if (T.desc.shape.size() != 3 || T.desc.stride.size() != 3) return Status::LayoutMismatch;

  const int64_t D0 = T.desc.shape[0];
  const int64_t D1 = T.desc.shape[1];
  const int64_t D2 = T.desc.shape[2];
  if (D0 <= 0 || D1 <= 0 || D2 <= 0) return Status::ShapeMismatch;

  // contiguous row-major check: [D0,D1,D2] with strides [D1*D2, D2, 1]
  const auto& s = T.desc.stride;
  if (!(s[2] == 1 && s[1] == D2 && s[0] == D1 * D2)) return Status::StrideMismatch;

  if (T.desc.dtype != DType::F32 && T.desc.dtype != DType::F16 && T.desc.dtype != DType::BF16)
    return Status::DtypeMismatch;

  return Status::Ok;
}


} // namespace ai
