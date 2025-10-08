#include <cuda_runtime.h>
#include "backends/cuda/ops/cross_entropy/api.hpp"

namespace ai {

static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

static inline bool is_row_major_2d_f32(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==2;
}
static inline bool is_row_major_1d_i32(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}

// 커널 런처 선언 (kernels.cu)
void ce_forward_logits_kernel_launcher(const float* X, const int32_t* T, float* loss_out,
                                       int M, int N, int ignore_index, float ls_eps,
                                       int reduction_kind, cudaStream_t s);
void ce_backward_logits_kernel_launcher(const float* X, const int32_t* T, float* dX,
                                        int M, int N, float inv_scale, int ignore_index, float ls_eps, cudaStream_t s);
void ce_forward_probs_kernel_launcher(const float* P, const int32_t* T, float* loss_out,
                                      int M, int N, int ignore_index, float eps, float ls_eps,
                                      int reduction_kind, cudaStream_t s);
void ce_backward_probs_kernel_launcher(const float* P, const int32_t* T, float* dX,
                                       int M, int N, float inv_scale, int ignore_index, float eps, float ls_eps, cudaStream_t s);

// ============================ Forward ============================
Status CrossEntropyCudaLaunch(const Tensor& X,
                              const Tensor& target,
                              Tensor& loss,
                              const CrossEntropyAttrs& attrs,
                              StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_1d_i32(target)) return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);
  if (static_cast<int>(target.desc.shape[0]) != M) return Status::ShapeMismatch;

  // loss shape 확인
  switch (attrs.reduction) {
    case Reduction::None:
      if (!(loss.device==Device::CUDA && loss.desc.dtype==DType::F32 &&
            loss.desc.layout==Layout::RowMajor && loss.desc.shape.size()==1 &&
            static_cast<int>(loss.desc.shape[0])==M))
        return Status::ShapeMismatch;
      break;
    case Reduction::Mean:
    case Reduction::Sum:
      if (!(loss.device==Device::CUDA && loss.desc.dtype==DType::F32 &&
            loss.desc.layout==Layout::RowMajor && loss.desc.shape.size()==1 &&
            static_cast<int>(loss.desc.shape[0])==1))
        return Status::ShapeMismatch;
      break;
    default:
      return Status::Invalid;
  }

  const float* Xp = static_cast<const float*>(X.data);
  const int32_t* Tp = static_cast<const int32_t*>(target.data);
  float* Lp = static_cast<float*>(loss.data);
  cudaStream_t s = to_cuda(stream);

  const int reduction_kind =
      (attrs.reduction==Reduction::None ? 0 :
       (attrs.reduction==Reduction::Mean ? 1 : 2));

  // Sum/Mean의 경우, loss[0]에 atomicAdd 누적 → 반드시 초기값 0 필요
  if (reduction_kind != 0) {
    auto e = cudaMemsetAsync(Lp, 0, sizeof(float), s);
    if (e != cudaSuccess) return Status::RuntimeError;
  }

  if (attrs.from_logits) {
    ce_forward_logits_kernel_launcher(Xp, Tp, Lp, M, N,
                                      attrs.ignore_index, attrs.ls_eps,
                                      reduction_kind, s);
  } else {
    // from_probs (확률)
    ce_forward_probs_kernel_launcher(Xp, Tp, Lp, M, N,
                                     attrs.ignore_index, attrs.eps, attrs.ls_eps,
                                     reduction_kind, s);
  }
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// ============================ Backward ============================
Status CrossEntropyCudaBackwardLaunch(const Tensor& X,
                                      const Tensor& target,
                                      Tensor& dX,
                                      const CrossEntropyAttrs& attrs,
                                      StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_1d_i32(target)) return Status::Invalid;
  if (!(dX.device==Device::CUDA && dX.desc.dtype==DType::F32 &&
        dX.desc.layout==Layout::RowMajor && dX.desc.shape.size()==2))
    return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);
  if (static_cast<int>(target.desc.shape[0]) != M) return Status::ShapeMismatch;
  if (dX.desc.shape[0] != X.desc.shape[0] || dX.desc.shape[1] != X.desc.shape[1])
    return Status::ShapeMismatch;

  const float* Xp = static_cast<const float*>(X.data);
  const int32_t* Tp = static_cast<const int32_t*>(target.data);
  float* dXp = static_cast<float*>(dX.data);

  // inv_scale: None/Sum -> 1, Mean -> 1/M (ignore_index는 분모에서 제외하지 않음)
  float inv_scale = 1.0f;
  if (attrs.reduction == Reduction::Mean) inv_scale = 1.0f / (float)M;

  if (attrs.from_logits) {
    ce_backward_logits_kernel_launcher(Xp, Tp, dXp, M, N,
                                       inv_scale, attrs.ignore_index, attrs.ls_eps,
                                       to_cuda(stream));
  } else {
    ce_backward_probs_kernel_launcher(Xp, Tp, dXp, M, N,
                                      inv_scale, attrs.ignore_index, attrs.eps, attrs.ls_eps,
                                      to_cuda(stream));
  }
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
