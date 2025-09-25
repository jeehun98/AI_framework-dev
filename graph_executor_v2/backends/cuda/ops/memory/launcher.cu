#include <cuda_runtime.h>
#include "backends/cuda/ops/memory/api.hpp"

namespace ai {

static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

template<int MAX_D=8>
__global__ void contiguous_copy_kernel(
  const float* __restrict__ x, float* __restrict__ y,
  int D,
  const int64_t* __restrict__ shape,
  const int64_t* __restrict__ stride_in,
  int64_t total)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;

  // tid -> 좌표
  int64_t idx[MAX_D];
  int64_t t = tid;
  #pragma unroll
  for (int d=D-1; d>=0; --d){
    idx[d] = t % shape[d];
    t /= shape[d];
  }

  // 좌표 -> 입력 오프셋
  int64_t off = 0;
  #pragma unroll
  for (int d=0; d<D; ++d){
    off += idx[d] * stride_in[d];
  }

  y[tid] = x[off];
}

Status ContiguousCopyCudaLaunch(const Tensor& X, Tensor& Y, StreamHandle stream)
{
  // 제약: F32, RowMajor, 1<=D<=8
  if (X.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return Status::Invalid;
  if (X.device != Device::CUDA || Y.device != Device::CUDA)     return Status::Invalid;
  if (X.desc.shape.size() != Y.desc.shape.size())               return Status::ShapeMismatch;

  const int D = (int)X.desc.shape.size();
  if (D < 1 || D > 8) return Status::Invalid;

  // shape 동일성만 보장(뷰/stride는 다를 수 있음)
  for (int i=0;i<D;++i){
    if (X.desc.shape[i] != Y.desc.shape[i]) return Status::ShapeMismatch;
  }

  // 메타 업로드
  int64_t h_shape[8], h_stride_in[8];
  int64_t total = 1;
  for (int i=0;i<D;++i){
    h_shape[i]     = X.desc.shape[i];
    h_stride_in[i] = X.desc.stride[i];
    total *= X.desc.shape[i];
  }

  int64_t *d_shape=nullptr, *d_stride_in=nullptr;
  cudaError_t e;
  e = cudaMalloc(&d_shape,     sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_stride_in, sizeof(int64_t)*D); if (e!=cudaSuccess){ cudaFree(d_shape); return Status::RuntimeError; }
  e = cudaMemcpyAsync(d_shape,     h_shape,     sizeof(int64_t)*D, cudaMemcpyHostToDevice, to_cuda(stream)); if (e!=cudaSuccess){ cudaFree(d_shape); cudaFree(d_stride_in); return Status::RuntimeError; }
  e = cudaMemcpyAsync(d_stride_in, h_stride_in, sizeof(int64_t)*D, cudaMemcpyHostToDevice, to_cuda(stream)); if (e!=cudaSuccess){ cudaFree(d_shape); cudaFree(d_stride_in); return Status::RuntimeError; }

  // 런치
  const int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1)/BS));
  contiguous_copy_kernel<8><<<grid, block, 0, to_cuda(stream)>>>(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    D, d_shape, d_stride_in, total
  );

  // 정리
  cudaFree(d_shape);
  cudaFree(d_stride_in);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
