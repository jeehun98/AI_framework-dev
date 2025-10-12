// backends/cuda/ops/concat/launcher.cu
#include <cuda_runtime.h>
#include <algorithm>
#include "backends/cuda/ops/concat/api.hpp"
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif

namespace ai {

static inline bool is_f32_cuda_row(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// kernel 선언 (kernels.cu)
void concat_copy_launcher(const float* const* in_ptrs,
                          const int64_t* sizes_axis,
                          int n_inputs,
                          float* out,
                          const int64_t* shape, int rank,
                          int axis,
                          cudaStream_t s);

Status ConcatCudaLaunch(const Tensor* inputs, int n_inputs, Tensor& output,
                        const ConcatAttrs& a, StreamHandle stream)
{
  if (n_inputs <= 0) return Status::Invalid;
  if (!is_f32_cuda_row(output)) return Status::Invalid;

  const int rank = (int)output.desc.shape.size();
  if (rank < 1 || rank > 4) return Status::Invalid;
  if (a.axis < 0 || a.axis >= rank) return Status::Invalid;

  // 기준 shape = output.shape, 단 axis 제외 동일 검증
  std::vector<int64_t> base = output.desc.shape;
  std::vector<int64_t> in_axis_sizes(n_inputs);
  std::vector<const float*> ptrs(n_inputs);

  for (int i=0; i<n_inputs; ++i){
    if (!is_f32_cuda_row(inputs[i])) return Status::Invalid;
    if ((int)inputs[i].desc.shape.size()!=rank) return Status::ShapeMismatch;
    for (int d=0; d<rank; ++d){
      if (d==a.axis) continue;
      if (inputs[i].desc.shape[d] != base[d]) return Status::ShapeMismatch;
    }
    in_axis_sizes[i] = inputs[i].desc.shape[a.axis];
    ptrs[i] = static_cast<const float*>(inputs[i].data);
  }

  // 출력 axis 합산 체크
  int64_t sum_axis = 0;
  for (auto v: in_axis_sizes) sum_axis += v;
  if (sum_axis != base[a.axis]) return Status::ShapeMismatch;

  auto s = to_cuda(stream);
  concat_copy_launcher(ptrs.data(), in_axis_sizes.data(), n_inputs,
                       static_cast<float*>(output.data), base.data(), rank, a.axis, s);
  return Status::Ok;
}

} // namespace ai
