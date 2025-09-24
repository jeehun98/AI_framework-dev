#include <cuda_runtime.h>
#include <algorithm>
#include "backends/cuda/ops/reduction/api.hpp"

namespace ai {

static inline bool is_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// 선언 (kernels.cu)
void make_rowmajor_stride(const std::vector<int64_t>& shape, std::vector<int64_t>& stride);
void normalize_axes(std::vector<int>& axes, int nd);
void launch_reduce_kernel(const float* X, float* Y,
                          const std::vector<int64_t>& kshape,
                          const std::vector<int64_t>& kstride,
                          const std::vector<int64_t>& rshape,
                          const std::vector<int64_t>& rstride,
                          ReduceOp op, cudaStream_t stream);

Status ReduceCudaLaunch(const Tensor& X, Tensor& Y,
                        const ReduceAttrs& attrs, StreamHandle stream)
{
  if (!is_f32_cuda(X) || !is_f32_cuda(Y)) return Status::Invalid;

  const auto& xshape = X.desc.shape;
  const int nd = (int)xshape.size();
  std::vector<int> axes = attrs.axes;
  normalize_axes(axes, nd);

  std::vector<int64_t> xstride;
  make_rowmajor_stride(xshape, xstride);

  // kept/reduced 분리
  std::vector<char> is_red(nd, 0);
  for (int a: axes) is_red[a] = 1;

  std::vector<int64_t> kshape, kstride, rshape, rstride;
  for (int i=0;i<nd;i++){
    if (is_red[i]) { rshape.push_back(xshape[i]); rstride.push_back(xstride[i]); }
    else           { kshape.push_back(xshape[i]); kstride.push_back(xstride[i]); }
  }
  if (kshape.empty()) { kshape = {1}; kstride = {0}; }     // all-reduce → output scalar [1]

  // 출력 요소수와 Y.shape 일치성은 바인딩에서 이미 구성했다고 가정(keepdim 처리 등)
  launch_reduce_kernel(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      kshape, kstride, rshape, rstride,
      attrs.op, to_cuda(stream));

  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
