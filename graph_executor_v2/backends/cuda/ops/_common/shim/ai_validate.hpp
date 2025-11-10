// backends/cuda/ops/_common/shim/ai_validate.hpp

#pragma once
#include "ai_status.hpp"
#include "ai_tensor.hpp"

namespace ai {

// ===== 기존 함수 유지 =====

// -------- 기초 체크 --------
inline Status expect_cuda(const Tensor& T) { return T.is_cuda()? Status::Ok : Status::DeviceMismatch; }
inline Status expect_layout(const Tensor& T, Layout L) { return (T.desc.layout==L)? Status::Ok : Status::LayoutMismatch; }
inline Status expect_rank(const Tensor& T, int k) { return (int)T.desc.shape.size()==k? Status::Ok : Status::ShapeMismatch; }
inline Status expect_dtype_in(const Tensor& T, std::initializer_list<DType> alw){
  for (auto dt: alw) if (T.desc.dtype==dt) return Status::Ok;
  return Status::DtypeMismatch;
}
inline Status expect_positive_numel(const Tensor& T){ return (T.numel()>0)? Status::Ok : Status::ShapeMismatch; }

// -------- 연속성 체크 (ND) --------
inline bool is_contiguous_rowmajor_nd(const Tensor& T) {
  if (T.desc.layout != Layout::RowMajor) return false;
  const auto& s  = T.desc.shape;
  const auto& st = T.desc.stride;
  if (s.size() != st.size()) return false;
  int64_t expect = 1;
  for (int i = (int)s.size() - 1; i >= 0; --i) {
    if (st[i] != expect) return false;
    // 음수/0 dim은 현재 정책상 허용하지 않음(아래 expect_positive_numel과 일치)
    expect *= s[i]; // int64 누적
  }
  return true;
}

// 범용 ND 검사: CUDA + RowMajor + 연속 + numel>0 + dtype∈allowed
inline Status expect_rowmajor_nd(const Tensor& T, int rank, std::initializer_list<DType> allowed) {
  AI_RETURN_IF_ERROR(expect_cuda(T));
  AI_RETURN_IF_ERROR(expect_layout(T, Layout::RowMajor));
  AI_RETURN_IF_ERROR(expect_rank(T, rank));
  if (!is_contiguous_rowmajor_nd(T)) return Status::StrideMismatch;
  AI_RETURN_IF_ERROR(expect_positive_numel(T));
  AI_RETURN_IF_ERROR(expect_dtype_in(T, allowed));
  return Status::Ok;
}

// 흔한 단축형
inline Status expect_rowmajor_4d(const Tensor& T) { return expect_rowmajor_nd(T, 4, {DType::F16, DType::F32, DType::BF16}); }
inline Status expect_rowmajor_1d(const Tensor& T) { return expect_rowmajor_nd(T, 1, {DType::F16, DType::F32, DType::BF16}); }

// -------- 모양/레이아웃 일치 --------
inline Status expect_same_shape_layout(const Tensor& A, const Tensor& B) {
  if (A.desc.layout != B.desc.layout) return Status::LayoutMismatch;
  if (A.desc.shape  != B.desc.shape)  return Status::ShapeMismatch;
  if (A.desc.stride != B.desc.stride) return Status::StrideMismatch;
  return Status::Ok;
}

inline Status expect_vec_len(const Tensor& T, int64_t C) {
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(T));
  return (T.desc.shape[0]==C)? Status::Ok : Status::ShapeMismatch;
}

// -------- alias 체크 --------
inline Status no_alias(const Tensor& A, const Tensor& B) {
  if (A.data == nullptr || B.data == nullptr) return Status::MissingInput;
  return (A.data != B.data)? Status::Ok : Status::Invalid;
}
// nullptr 허용 버전(한쪽/양쪽 nullptr면 Ok로 간주)
inline Status no_alias_allow_null(const Tensor* A, const Tensor* B) {
  if (!A || !B) return Status::Ok;
  if (!A->data || !B->data) return Status::Ok;
  return (A->data != B->data)? Status::Ok : Status::Invalid;
}

// -------- dtype 조합 --------
inline Status expect_io_mixed_f16_f32(const Tensor& T) { return expect_dtype_in(T, {DType::F16, DType::F32}); }
inline Status expect_param_f32(const Tensor& T) { return (T.desc.dtype==DType::F32)? Status::Ok : Status::DtypeMismatch; }

// -------- axis 정규화 --------
inline Status normalize_axis(int axis, int rank, /*out*/int& out_axis) {
  if (rank <= 0) return Status::InvalidArgument;
  int a = axis;
  if (a < 0) a += rank;
  if (a < 0 || a >= rank) return Status::InvalidArgument;
  out_axis = a;
  return Status::Ok;
}

// -------- NCHW/NHWC 파싱 --------
struct NCHW4 { int N, C, H, W; };
inline Status get_dims_4d(const Tensor& X, bool channels_last, NCHW4& out) {
  AI_RETURN_IF_ERROR(expect_rowmajor_4d(X));
  const auto& s = X.desc.shape;
  out = channels_last ? NCHW4{(int)s[0], (int)s[3], (int)s[1], (int)s[2]}
                      : NCHW4{(int)s[0], (int)s[1], (int)s[2], (int)s[3]};
  if (out.N<=0 || out.C<=0 || out.H<=0 || out.W<=0) return Status::ShapeMismatch;
  return Status::Ok;
}

// -------- 다수 텐서 공통 차원 검사(Concat 등) --------
inline Status expect_same_except_axis(const std::vector<Tensor>& Xs, int rank, int axis_norm) {
  if (Xs.empty()) return Status::MissingInput;
  const auto& ref = Xs[0];
  AI_RETURN_IF_ERROR(expect_rowmajor_nd(ref, rank, {ref.desc.dtype})); // dtype은 동일성만 체킹

  for (size_t i=1;i<Xs.size();++i){
    const auto& t = Xs[i];
    if (t.desc.layout != ref.desc.layout) return Status::LayoutMismatch;
    if ((int)t.desc.shape.size()!=rank) return Status::ShapeMismatch;
    for (int d=0; d<rank; ++d){
      if (d==axis_norm) continue;
      if (t.desc.shape[d] != ref.desc.shape[d]) return Status::ShapeMismatch;
    }
  }
  return Status::Ok;
}

[[nodiscard]] inline bool is_cuda_f32_rowmajor(const ai::Tensor& T) noexcept {
  return T.is_cuda()
      && T.desc.dtype  == ai::DType::F32
      && T.desc.layout == ai::Layout::RowMajor
      && T.desc.shape.size() == 2;
}

} // namespace ai
