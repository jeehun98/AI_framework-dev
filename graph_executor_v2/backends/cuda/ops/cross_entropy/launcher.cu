// backends/cuda/ops/cross_entropy/launcher.cu
#include <cuda_runtime.h>
#include <vector>
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
         (t.desc.dtype==DType::I32);
}

static inline cudaStream_t to_cuda(StreamHandle h){
  return reinterpret_cast<cudaStream_t>(h);
}

// CUDA 커널 런처 시그니처 (int32 인덱스)
void ce_forward_logits_kernel_launcher(const float* X,
                                       const int32_t* T,
                                       float* loss_vec,  // [M] per-sample
                                       int M, int N,
                                       cudaStream_t s);

void ce_backward_logits_kernel_launcher(const float* X,
                                        const int32_t* T,
                                        float* dX,
                                        int M, int N,
                                        bool mean_reduction,
                                        cudaStream_t s);

Status CrossEntropyCudaLaunch(const Tensor& X,
                              const Tensor& target,
                              Tensor& loss,
                              const CrossEntropyAttrs& attrs,
                              StreamHandle stream)
{
  // 상세 검증: 장치/ dtype/ 레이아웃/ 차원/ shape
  if (X.device!=Device::CUDA || target.device!=Device::CUDA || loss.device!=Device::CUDA)
    return Status::DeviceMismatch;

  if (X.desc.dtype!=DType::F32 || loss.desc.dtype!=DType::F32)
    return Status::DtypeMismatch;

  if (X.desc.layout!=Layout::RowMajor || target.desc.layout!=Layout::RowMajor || loss.desc.layout!=Layout::RowMajor)
    return Status::LayoutMismatch;

  if (X.desc.shape.size()!=2) return Status::ShapeMismatch;
  if (target.desc.shape.size()!=1) return Status::ShapeMismatch;
  if (loss.desc.shape.size()!=1) return Status::ShapeMismatch;

  if (target.desc.dtype!=DType::I32)
    return Status::DtypeMismatch;

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
    // per-sample 손실을 직접 loss 버퍼에 작성
    float* Lp = static_cast<float*>(loss.data);
    ce_forward_logits_kernel_launcher(Xp, Tp, Lp, M, N, s);
    // 커널 런칭 오류는 런타임 동기화 시점에서 잡힘
    if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
    return Status::Ok;
  } else {
    // reduction != None: 임시 device 벡터에 per-sample 계산 → Host에서 reduce → 스칼라를 Device로 복사
    float* dLossVec = nullptr;
    if (cudaMalloc(&dLossVec, sizeof(float)*M) != cudaSuccess) return Status::RuntimeError;

    ce_forward_logits_kernel_launcher(Xp, Tp, dLossVec, M, N, s);
    if (cudaPeekAtLastError()!=cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }

    std::vector<float> host(M);
    if (cudaMemcpyAsync(host.data(), dLossVec, sizeof(float)*M, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
      cudaFree(dLossVec); return Status::RuntimeError;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }

    double acc = 0.0;
    for (int i=0;i<M;++i) acc += host[i];
    if (attrs.reduction == Reduction::Mean) acc /= static_cast<double>(M);
    float out = static_cast<float>(acc);

    float* Lp_scalar = static_cast<float*>(loss.data);
    if (cudaMemcpyAsync(Lp_scalar, &out, sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) {
      cudaFree(dLossVec); return Status::RuntimeError;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) { cudaFree(dLossVec); return Status::RuntimeError; }

    cudaFree(dLossVec);
    return Status::Ok;
  }
}

Status CrossEntropyCudaBackwardLaunch(const Tensor& X,
                                      const Tensor& target,
                                      Tensor& dX,
                                      const CrossEntropyAttrs& attrs,
                                      StreamHandle stream)
{
  // 상세 검증
  if (X.device!=Device::CUDA || target.device!=Device::CUDA || dX.device!=Device::CUDA)
    return Status::DeviceMismatch;

  if (X.desc.dtype!=DType::F32 || dX.desc.dtype!=DType::F32)
    return Status::DtypeMismatch;

  if (X.desc.layout!=Layout::RowMajor || target.desc.layout!=Layout::RowMajor || dX.desc.layout!=Layout::RowMajor)
    return Status::LayoutMismatch;

  if (X.desc.shape.size()!=2 || dX.desc.shape.size()!=2) return Status::ShapeMismatch;
  if (target.desc.shape.size()!=1) return Status::ShapeMismatch;
  if (X.desc.shape != dX.desc.shape) return Status::ShapeMismatch;

  if (target.desc.dtype!=DType::I32)
    return Status::DtypeMismatch;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const float*   Xp  = static_cast<const float*>(X.data);
  float*         dXp = static_cast<float*>(dX.data);
  const int32_t* Tp  = static_cast<const int32_t*>(target.data);
  cudaStream_t   s   = to_cuda(stream);

  const bool mean_red = (attrs.reduction == Reduction::Mean);
  ce_backward_logits_kernel_launcher(Xp, Tp, dXp, M, N, mean_red, s);
  if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;

  return Status::Ok;
}

} // namespace ai
