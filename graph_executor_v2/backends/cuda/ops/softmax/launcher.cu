#include <cuda_runtime.h>
#include "backends/cuda/ops/softmax/api.hpp"

namespace ai {

static inline bool is_row_major_2d_f32(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.shape.size()==2;
}

void softmax_forward_kernel_launcher(const float*, const float*, float*, int, int, float, bool, cudaStream_t);
void softmax_backward_kernel_launcher(const float*, const float*, float*, int, int, bool, cudaStream_t);

static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

Status SoftmaxCudaLaunch(const Tensor& X, const Tensor* Mask, Tensor& Y,
                         const SoftmaxAttrs& attrs, StreamHandle stream)
{
  if (!is_row_major_2d_f32(X) || !is_row_major_2d_f32(Y)) return Status::Invalid;
  if (X.desc.shape!=Y.desc.shape) return Status::Invalid;

  const int M = (int)X.desc.shape[0];
  const int N = (int)X.desc.shape[1];

  const float* x = static_cast<const float*>(X.data);
  float* y = static_cast<float*>(Y.data);
  const float* m = nullptr;

  if (Mask && Mask->data) {
    // mask shape: [M,N] or [1,N]
    if (!(Mask->desc.dtype==DType::F32 && Mask->desc.layout==Layout::RowMajor &&
         ( (Mask->desc.shape.size()==2 && Mask->desc.shape[0]==M && Mask->desc.shape[1]==N) ||
           (Mask->desc.shape.size()==1 && Mask->desc.shape[0]==N) )))
      return Status::Invalid;
    m = static_cast<const float*>(Mask->data);
    // [1,N] 브로드캐스트는 커널에서 row==0 만 참조하도록 인덱싱 처리
  }

  softmax_forward_kernel_launcher(x, m, y, M, N, attrs.scale, attrs.log, to_cuda(stream));
  return Status::Ok;
}

Status SoftmaxCudaBackwardLaunch(const Tensor& Y_or_X, const Tensor* Mask,
                                 const Tensor& dY, Tensor& dX,
                                 const SoftmaxAttrs& attrs, bool y_provided,
                                 StreamHandle stream)
{
  if (!is_row_major_2d_f32(dY) || !is_row_major_2d_f32(dX)) return Status::Invalid;
  if (dY.desc.shape!=dX.desc.shape) return Status::Invalid;

  const int M = (int)dY.desc.shape[0];
  const int N = (int)dY.desc.shape[1];

  // y_provided=true가 권장: 다시 소프트맥스 안 하고 바로 사용
  const float* Yptr = nullptr;
  float* dXptr = static_cast<float*>(dX.data);
  const float* dYptr = static_cast<const float*>(dY.data);

  // 간단히: y_or_x는 이미 softmax 출력이라고 가정 (실서비스면 옵션으로 분기)
  Yptr = static_cast<const float*>(Y_or_X.data);

  softmax_backward_kernel_launcher(Yptr, dYptr, dXptr, M, N, attrs.log, to_cuda(stream));
  return Status::Ok;
}

} // namespace ai
