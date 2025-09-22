// backends/cuda/ops/cross_entropy/launcher.cu
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "backends/cuda/ops/cross_entropy/api.hpp"

namespace ai {

static inline bool is_rowmajor_2d_f32(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}
static inline bool is_1d_index_i32(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==1 &&
         t.desc.dtype==DType::I32;
}
static inline bool is_1d_f32(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==1 &&
         t.desc.dtype==DType::F32;
}
static inline cudaStream_t to_cuda(StreamHandle h){
  return reinterpret_cast<cudaStream_t>(h);
}

// === 커널 런처 시그니처(선언/정의/호출 동일 유지) ===
void ce_forward_logits_kernel_launcher(const float* X, const int32_t* T,
                                       float* loss_vec, int M, int N,
                                       int ignore_index, float ls_eps,
                                       cudaStream_t s);

void ce_backward_logits_kernel_launcher(const float* X, const int32_t* T,
                                        float* dX, int M, int N,
                                        float inv_scale,
                                        int ignore_index, float ls_eps,
                                        cudaStream_t s);

// ========================= Forward =========================
Status CrossEntropyCudaLaunch(const Tensor& X,
                              const Tensor& target,
                              Tensor& loss,
                              const CrossEntropyAttrs& attrs,
                              StreamHandle stream)
{
  // 형식/장치/레이아웃/차원 검증
  if (!is_rowmajor_2d_f32(X) || !is_1d_index_i32(target) || !is_1d_f32(loss))
    return Status::Invalid;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const bool want_vec = (attrs.reduction == Reduction::None);
  if (want_vec) {
    if (loss.desc.shape[0] != M) return Status::ShapeMismatch;
  } else {
    if (loss.desc.shape[0] != 1) return Status::ShapeMismatch;
  }

  const float*   Xp = static_cast<const float*>(X.data);
  const int32_t* Tp = static_cast<const int32_t*>(target.data);
  cudaStream_t   s  = to_cuda(stream);

  if (want_vec) {
    // per-sample loss in-place
    float* Lp = static_cast<float*>(loss.data);
    ce_forward_logits_kernel_launcher(Xp, Tp, Lp, M, N,
                                      /*ignore_index*/ attrs.ignore_index,
                                      /*ls_eps*/       attrs.ls_eps,
                                      s);
    if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
    return Status::Ok;
  } else {
    // 1) per-sample 손실을 임시 device 벡터에 계산
    float* dLossVec = nullptr;
    if (cudaMalloc(&dLossVec, sizeof(float)*M) != cudaSuccess) return Status::RuntimeError;

    ce_forward_logits_kernel_launcher(Xp, Tp, dLossVec, M, N,
                                      /*ignore_index*/ attrs.ignore_index,
                                      /*ls_eps*/       attrs.ls_eps,
                                      s);
    if (cudaPeekAtLastError()!=cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }

    // 2) Host로 가져와 리덕션 (Mean은 Meff(유효 샘플 수)로 나눔)
    std::vector<float>   host_loss(M);
    std::vector<int32_t> host_t(M);

    if (cudaMemcpyAsync(host_loss.data(), dLossVec, sizeof(float)*M, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
      cudaFree(dLossVec); return Status::RuntimeError;
    }
    if (cudaMemcpyAsync(host_t.data(), Tp, sizeof(int32_t)*M, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
      cudaFree(dLossVec); return Status::RuntimeError;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }

    int Meff = 0;
    for (int i=0;i<M;++i) if (host_t[i] != attrs.ignore_index) ++Meff;
    if (Meff == 0) Meff = 1; // all ignored → mean 0을 안전히 표현하기 위해 분모 보호

    double acc = 0.0;
    for (int i=0;i<M;++i) acc += host_loss[i];

    float out = 0.f;
    if (attrs.reduction == Reduction::Sum) {
      out = static_cast<float>(acc);
    } else { // Mean
      out = static_cast<float>(acc / static_cast<double>(Meff));
    }

    float* Lp_scalar = static_cast<float*>(loss.data);
    if (cudaMemcpyAsync(Lp_scalar, &out, sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) {
      cudaFree(dLossVec); return Status::RuntimeError;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }
    cudaFree(dLossVec);
    return Status::Ok;
  }
}

// ========================= Backward =========================
Status CrossEntropyCudaBackwardLaunch(const Tensor& X,
                                      const Tensor& target,
                                      Tensor& dX,
                                      const CrossEntropyAttrs& attrs,
                                      StreamHandle stream)
{
  if (!is_rowmajor_2d_f32(X) || !is_rowmajor_2d_f32(dX) || !is_1d_index_i32(target))
    return Status::Invalid;
  if (X.desc.shape != dX.desc.shape) return Status::ShapeMismatch;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const float*   Xp = static_cast<const float*>(X.data);
  const int32_t* Tp = static_cast<const int32_t*>(target.data);
  float*         dXp= static_cast<float*>(dX.data);
  cudaStream_t   s  = to_cuda(stream);

  // inv_scale 계산: Mean이면 1/Meff, Sum/None이면 1
  float inv_scale = 1.f;
  if (attrs.reduction == Reduction::Mean) {
    std::vector<int32_t> host_t(M);
    if (cudaMemcpyAsync(host_t.data(), Tp, sizeof(int32_t)*M, cudaMemcpyDeviceToHost, s) != cudaSuccess)
      return Status::RuntimeError;
    if (cudaStreamSynchronize(s) != cudaSuccess) return Status::RuntimeError;

    int Meff = 0;
    for (int i=0;i<M;++i) if (host_t[i] != attrs.ignore_index) ++Meff;
    inv_scale = (Meff > 0) ? (1.f / static_cast<float>(Meff)) : 0.f;
  }

  ce_backward_logits_kernel_launcher(Xp, Tp, dXp, M, N,
                                     /*inv_scale*/    inv_scale,
                                     /*ignore_index*/ attrs.ignore_index,
                                     /*ls_eps*/       attrs.ls_eps,
                                     s);
  if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

} // namespace ai
