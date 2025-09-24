#include <cuda_runtime.h>
#include <stdint.h>
#include "backends/cuda/ops/pad/api.hpp"

namespace ai {

static inline bool is_rowmajor_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// ---- device helpers ----
template<int MAXD>
__device__ __forceinline__
void linear_to_coords(int64_t idx, const int64_t* shape, int D, int64_t* coord) {
  #pragma unroll
  for (int i = D - 1; i >= 0; --i) { coord[i] = idx % shape[i]; idx /= shape[i]; }
}

template<int MAXD>
__device__ __forceinline__
int64_t coords_to_offset(const int64_t* coord, const int64_t* stride, int D) {
  int64_t off = 0;
  #pragma unroll
  for (int i=0;i<D;++i) off += coord[i] * stride[i];
  return off;
}

// ---- kernels ----
template<int MAXD=8>
__global__ void pad_constant_kernel(
  const float* __restrict__ x, float* __restrict__ y,
  const int64_t* __restrict__ ishape, const int64_t* __restrict__ istride,
  const int64_t* __restrict__ oshape, const int64_t* __restrict__ ostride, // (contig 가정이지만 받아둠)
  const int*     __restrict__ before,
  int D, float value, int64_t total_out)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_out) return;

  int64_t ocoord[MAXD];
  linear_to_coords<MAXD>(tid, oshape, D, ocoord);

  // map to input coord
  bool inside = true;
  int64_t icoord[MAXD];
  #pragma unroll
  for (int i=0;i<D;++i){
    int64_t xi = ocoord[i] - (int64_t)before[i];
    icoord[i] = xi;
    if (xi < 0 || xi >= ishape[i]) { inside=false; }
  }

  const int64_t y_off = coords_to_offset<MAXD>(ocoord, ostride, D);
  if (!inside) {
    y[y_off] = value;
  } else {
    const int64_t x_off = coords_to_offset<MAXD>(icoord, istride, D);
    y[y_off] = __ldg(x + x_off);
  }
}

template<int MAXD=8>
__global__ void pad_backward_kernel( // copy dY (unpadded slice) into dX
  const float* __restrict__ dy, float* __restrict__ dx,
  const int64_t* __restrict__ ishape, const int64_t* __restrict__ istride,  // in (X/dX)
  const int64_t* __restrict__ oshape, const int64_t* __restrict__ ostride,  // out (Y/dY)
  const int*     __restrict__ before,
  int D, int64_t total_in)
{
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_in) return;

  int64_t icoord[MAXD];
  linear_to_coords<MAXD>(tid, ishape, D, icoord);

  // find corresponding coord in dY: yi = xi + before[i]
  int64_t ocoord[MAXD];
  #pragma unroll
  for (int i=0;i<D;++i) ocoord[i] = icoord[i] + (int64_t)before[i];

  const int64_t x_off = coords_to_offset<MAXD>(icoord, istride, D);
  const int64_t y_off = coords_to_offset<MAXD>(ocoord, ostride, D);
  dx[x_off] = dy[y_off];
}

// ---- launchers ----
static inline size_t vec_prod(const std::vector<int64_t>& v){
  size_t p = 1; for (auto x : v) p *= (size_t)x; return p;
}

Status PadCudaLaunch(const Tensor& X, Tensor& Y, const PadSpec& s, StreamHandle stream)
{
  if (!is_rowmajor_f32_cuda(X) || !is_rowmajor_f32_cuda(Y)) return Status::Invalid;
  const int D = (int)X.desc.shape.size();
  if ((int)Y.desc.shape.size() != D) return Status::ShapeMismatch;
  if ((int)s.before.size()!=D || (int)s.after.size()!=D) return Status::Invalid;

  // sanity on shapes: Y.shape[d] == X.shape[d] + before[d] + after[d]
  for (int i=0;i<D;++i){
    if (Y.desc.shape[i] != X.desc.shape[i] + s.before[i] + s.after[i])
      return Status::ShapeMismatch;
  }

  // Upload meta to device
  int64_t *d_ishape=nullptr, *d_istride=nullptr, *d_oshape=nullptr, *d_ostride=nullptr;
  int *d_before=nullptr;
  cudaError_t e;
  e = cudaMalloc(&d_ishape,  sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_istride, sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_oshape,  sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_ostride, sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_before,  sizeof(int)*D);     if (e!=cudaSuccess) return Status::RuntimeError;

  e = cudaMemcpy(d_ishape,  X.desc.shape.data(),    sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_istride, X.desc.stride.data(),   sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_oshape,  Y.desc.shape.data(),    sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_ostride, Y.desc.stride.data(),   sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_before,  s.before.data(),        sizeof(int)*D,     cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;

  // launch
  const int64_t total_out = (int64_t)vec_prod(Y.desc.shape);
  const int BS = 256;
  dim3 block(BS), grid((unsigned)((total_out + BS - 1)/BS));
  pad_constant_kernel<<<grid, block, 0, to_cuda(stream)>>>(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      d_ishape, d_istride, d_oshape, d_ostride, d_before,
      D, s.value, total_out
  );
  e = cudaPeekAtLastError();
  cudaFree(d_ishape); cudaFree(d_istride); cudaFree(d_oshape); cudaFree(d_ostride); cudaFree(d_before);
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status PadBackwardCudaLaunch(const Tensor& dY, Tensor& dX, const PadSpec& s, StreamHandle stream)
{
  if (!is_rowmajor_f32_cuda(dY) || !is_rowmajor_f32_cuda(dX)) return Status::Invalid;
  const int D = (int)dX.desc.shape.size();
  if ((int)dY.desc.shape.size() != D) return Status::ShapeMismatch;
  if ((int)s.before.size()!=D || (int)s.after.size()!=D) return Status::Invalid;

  // dY.shape[d] must match dX.shape[d] + pads
  for (int i=0;i<D;++i){
    if (dY.desc.shape[i] != dX.desc.shape[i] + s.before[i] + s.after[i])
      return Status::ShapeMismatch;
  }

  int64_t *d_ishape=nullptr, *d_istride=nullptr, *d_oshape=nullptr, *d_ostride=nullptr;
  int *d_before=nullptr;
  cudaError_t e;
  e = cudaMalloc(&d_ishape,  sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_istride, sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_oshape,  sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_ostride, sizeof(int64_t)*D); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMalloc(&d_before,  sizeof(int)*D);     if (e!=cudaSuccess) return Status::RuntimeError;

  e = cudaMemcpy(d_ishape,  dX.desc.shape.data(),  sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_istride, dX.desc.stride.data(), sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_oshape,  dY.desc.shape.data(),  sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_ostride, dY.desc.stride.data(), sizeof(int64_t)*D, cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;
  e = cudaMemcpy(d_before,  s.before.data(),       sizeof(int)*D,     cudaMemcpyHostToDevice); if (e!=cudaSuccess) return Status::RuntimeError;

  const int64_t total_in = (int64_t)vec_prod(dX.desc.shape);
  const int BS = 256;
  dim3 block(BS), grid((unsigned)((total_in + BS - 1)/BS));
  pad_backward_kernel<<<grid, block, 0, to_cuda(stream)>>>(
      static_cast<const float*>(dY.data),
      static_cast<float*>(dX.data),
      d_ishape, d_istride, d_oshape, d_ostride, d_before,
      D, total_in
  );
  e = cudaPeekAtLastError();
  cudaFree(d_ishape); cudaFree(d_istride); cudaFree(d_oshape); cudaFree(d_ostride); cudaFree(d_before);
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
