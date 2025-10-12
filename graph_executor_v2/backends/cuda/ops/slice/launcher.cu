// backends/cuda/ops/slice/launcher.cu
#include <cuda_runtime.h>
#include "backends/cuda/ops/slice/api.hpp"
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif

namespace ai {
static inline bool is_f32_row(const Tensor&t){return t.device==Device::CUDA&&t.desc.dtype==DType::F32&&t.desc.layout==Layout::RowMajor;}
static inline cudaStream_t to_cuda(StreamHandle h){return reinterpret_cast<cudaStream_t>(h);}

// kernel 선언
void slice_copy_launcher(const float* x, float* y,
                         const int64_t* xshape, const int64_t* yshape,
                         const int* starts, int rank, cudaStream_t s);

Status SliceCudaLaunch(const Tensor& X, Tensor& Y, const SliceAttrs& a, StreamHandle stream)
{
  if (!is_f32_row(X) || !is_f32_row(Y)) return Status::Invalid;
  if (a.rank<1 || a.rank>4) return Status::Invalid;
  if ((int)X.desc.shape.size()!=a.rank || (int)Y.desc.shape.size()!=a.rank) return Status::ShapeMismatch;

  for (int d=0; d<a.rank; ++d){
    if (a.starts[d] < 0) return Status::Invalid;
    if (a.starts[d] + a.sizes[d] > X.desc.shape[d]) return Status::ShapeMismatch;
    if (Y.desc.shape[d] != a.sizes[d]) return Status::ShapeMismatch;
  }
  auto s = to_cuda(stream);
  slice_copy_launcher((const float*)X.data, (float*)Y.data,
                      X.desc.shape.data(), Y.desc.shape.data(),
                      a.starts, a.rank, s);
  return Status::Ok;
}
} // namespace ai
