// backends/cuda/ops/embedding/launcher.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "backends/cuda/ops/embedding/api.hpp"
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif

namespace ai {

static inline bool is_i32_cuda_1or2(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.layout==Layout::RowMajor &&
         t.desc.dtype==DType::I32 &&
         (t.desc.shape.size()==1 || t.desc.shape.size()==2);
}
static inline bool is2_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         (t.desc.shape.size()==2);
}
static inline bool is3_or2_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA &&
         t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor &&
         (t.desc.shape.size()==2 || t.desc.shape.size()==3);
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ---- kernels (I32 고정) ----
void embedding_forward_launcher(
  const float* W, int V, int D,
  const int*  I, int N, int L,
  int padding_idx, float out_scale,
  float* Y, bool y_is_3d,
  cudaStream_t s);

void embedding_backward_scatter_launcher(
  const int* I, int N, int L,
  const float* dY, bool dy_is_3d,
  int V, int D,
  int padding_idx,
  const int* freq /*nullable*/, bool scale_grad_by_freq,
  float* dW,
  cudaStream_t s);

void count_frequency_launcher(
  const int* I, int N, int L, int V,
  int* out_freq, cudaStream_t s);

Status EmbeddingCudaLaunch(
  const Tensor& Weight, const Tensor& Indices, Tensor& Output,
  const EmbeddingAttrs& attrs, StreamHandle stream)
{
  if (!is2_f32_cuda(Weight)) return Status::Invalid;
  const int V = (int)Weight.desc.shape[0];
  const int D = (int)Weight.desc.shape[1];

  // I64 미지원: 반드시 I32
  if (!is_i32_cuda_1or2(Indices)) return Status::Invalid;

  int N=1, L=0;
  if (Indices.desc.shape.size()==2) { N=(int)Indices.desc.shape[0]; L=(int)Indices.desc.shape[1]; }
  else { L=(int)Indices.desc.shape[0]; }

  // Output shape check
  bool y_is_3d=false;
  if (Output.desc.shape.size()==3) { y_is_3d=true;
    if ((int)Output.desc.shape[0]!=N || (int)Output.desc.shape[1]!=L || (int)Output.desc.shape[2]!=D) return Status::ShapeMismatch;
  } else if (Output.desc.shape.size()==2) {
    if (N!=1 || (int)Output.desc.shape[0]!=L || (int)Output.desc.shape[1]!=D) return Status::ShapeMismatch;
  } else return Status::Invalid;

  if (attrs.padding_idx >= 0 && attrs.padding_idx >= V) return Status::Invalid;

  auto s = to_cuda(stream);
  embedding_forward_launcher(
    (const float*)Weight.data, V, D,
    (const int*)Indices.data, N, L,
    attrs.padding_idx, attrs.out_scale,
    (float*)Output.data, y_is_3d, s);

  return Status::Ok;
}

Status EmbeddingCudaBackwardLaunch(
  const Tensor& Indices, const Tensor& dY, Tensor* dWeight,
  const EmbeddingAttrs& attrs, StreamHandle stream)
{
  if (!dWeight) return Status::Ok;
  if (!is2_f32_cuda(*dWeight)) return Status::Invalid;
  const int V = (int)dWeight->desc.shape[0];
  const int D = (int)dWeight->desc.shape[1];

  if (!is_i32_cuda_1or2(Indices)) return Status::Invalid;

  int N=1, L=0;
  if (Indices.desc.shape.size()==2) { N=(int)Indices.desc.shape[0]; L=(int)Indices.desc.shape[1]; }
  else { L=(int)Indices.desc.shape[0]; }

  if (!is3_or2_f32_cuda(dY)) return Status::Invalid;
  bool dy_is_3d = (dY.desc.shape.size()==3);
  if (dy_is_3d) {
    if ((int)dY.desc.shape[0]!=N || (int)dY.desc.shape[1]!=L || (int)dY.desc.shape[2]!=D) return Status::ShapeMismatch;
  } else {
    if (N!=1 || (int)dY.desc.shape[0]!=L || (int)dY.desc.shape[1]!=D) return Status::ShapeMismatch;
  }

  if (attrs.padding_idx >= 0 && attrs.padding_idx >= V) return Status::Invalid;

  auto s = to_cuda(stream);

  const int* freq_dev = nullptr; // (옵션) 빈도 스케일 미구현
  embedding_backward_scatter_launcher(
    (const int*)Indices.data, N, L,
    (const float*)dY.data, dy_is_3d,
    V, D, attrs.padding_idx, freq_dev, attrs.scale_grad_by_freq,
    (float*)dWeight->data, s);

  return Status::Ok;
}

} // namespace ai
