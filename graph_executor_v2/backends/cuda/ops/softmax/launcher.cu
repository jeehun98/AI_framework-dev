#include <cuda_runtime.h>
#include "backends/cuda/ops/softmax/api.hpp"

namespace ai {

static inline bool is_row_major_2d_f32(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}

static inline bool is_row_major_2d_or_vec_f32(const Tensor& t){
  // 허용: [M,N], [1,N], [M,1], [N]
  if (t.device!=Device::CUDA || t.desc.dtype!=DType::F32 || t.desc.layout!=Layout::RowMajor)
    return false;
  const auto R = t.desc.shape.size();
  return (R==1 || R==2);
}

static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ---------------- 커널 런처 시그니처 (mask_kind 추가) ----------------
void softmax_forward_kernel_launcher(const float* x,
                                     const float* mask,   // null 가능
                                     float* y,
                                     int M, int N,
                                     float scale,
                                     bool log_softmax,
                                     int mask_kind,       // 0=None, 1=[M,N], 2=[1,N]/[N], 3=[M,1]
                                     cudaStream_t s);

void softmax_backward_kernel_launcher(const float* y,     // softmax or log_softmax output
                                      const float* gy,    // dY
                                      float* gx,          // dX
                                      int M, int N,
                                      float scale,
                                      bool log_softmax,
                                      cudaStream_t s);

// ---------------- 마스크 kind 판별 ----------------
static inline int mask_kind_of(const Tensor* Mask, int M, int N){
  if (!Mask || !Mask->data) return 0;
  const auto& ms = Mask->desc.shape;
  if (ms.size()==2){
    if (ms[0]==M && ms[1]==N) return 1;   // [M,N]
    if (ms[0]==1 && ms[1]==N) return 2;   // [1,N]
    if (ms[0]==M && ms[1]==1) return 3;   // [M,1]
  } else if (ms.size()==1){
    if (ms[0]==N) return 2;               // [N] -> [1,N]
  }
  return -1; // invalid
}

// ============================ Forward ============================
Status SoftmaxCudaLaunch(const Tensor& X,
                         const Tensor* Mask,     // null 가능
                         Tensor& Y,
                         const SoftmaxAttrs& attrs,
                         StreamHandle stream,
                         const SoftmaxWorkspaceFwd* /*ws_fwd*/)  // 현재 커널은 WS 미사용
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y))
    return Status::Invalid;
  if (X.desc.shape != Y.desc.shape)
    return Status::ShapeMismatch;

  const int M = static_cast<int>(X.desc.shape[0]);
  const int N = static_cast<int>(X.desc.shape[1]);

  const float* x = static_cast<const float*>(X.data);
  float* y       = static_cast<float*>(Y.data);

  const float* mptr = nullptr;
  int mk = 0;
  if (Mask && Mask->data){
    if (!is_row_major_2d_or_vec_f32(*Mask)) return Status::Invalid;
    mk = mask_kind_of(Mask, M, N);
    if (mk < 0) return Status::ShapeMismatch;
    mptr = static_cast<const float*>(Mask->data);
  }

  softmax_forward_kernel_launcher(
      x, mptr, y,
      M, N, attrs.scale, attrs.log,
      mk,                            // ★ mask_kind 전달
      to_cuda(stream)
  );
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// ============================ Backward ============================
Status SoftmaxCudaBackwardLaunch(const Tensor& Y_or_X,   // Y 권장
                                 const Tensor* /*Mask*/, // bwd에선 마스크 미사용
                                 const Tensor& dY,
                                 Tensor& dX,
                                 const SoftmaxAttrs& attrs,
                                 bool y_provided,
                                 StreamHandle stream,
                                 const SoftmaxWorkspaceBwd* /*ws_bwd*/)  // 현재 커널은 WS 미사용
{
  if (!is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX))
    return Status::Invalid;
  if (dY.desc.shape != dX.desc.shape)
    return Status::ShapeMismatch;

  if (!is_row_major_2d_f32(Y_or_X)) return Status::Invalid;
  if (Y_or_X.desc.shape != dY.desc.shape) return Status::ShapeMismatch;
  if (!y_provided) {
    // 재계산 커널 미제공: 현재는 Y 필요
    return Status::Invalid;
  }

  const int M = static_cast<int>(dY.desc.shape[0]);
  const int N = static_cast<int>(dY.desc.shape[1]);

  const float* y  = static_cast<const float*>(Y_or_X.data); // forward 출력 (softmax 또는 log-softmax)
  const float* gy = static_cast<const float*>(dY.data);
  float* gx       = static_cast<float*>(dX.data);

  softmax_backward_kernel_launcher(
      y, gy, gx,
      M, N, attrs.scale, attrs.log,
      to_cuda(stream)
  );
  auto err = cudaPeekAtLastError();
  return (err==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
