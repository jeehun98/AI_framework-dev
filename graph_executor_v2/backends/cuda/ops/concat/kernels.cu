#include <cuda_runtime.h>
#include <cstdint>

// 공통: 임의 region 복사/가산 커널
namespace {

  template<int BS>
  __global__ void tensor_copy_region_kernel(
      const float* __restrict__ X,
      float* __restrict__ Y,
      int rank,
      const int* __restrict__ reg_dims,
      const int* __restrict__ x_strides,
      const int* __restrict__ y_strides,
      const int* __restrict__ x_starts,
      const int* __restrict__ y_starts)
  {
    // total elements in the region
    int64_t total = 1;
    #pragma unroll
    for (int d=0; d<4; ++d) if (d<rank) total *= (int64_t)reg_dims[d];
    if (total <= 0) return;

    // region row-major strides
    int rstr[4] = {1,1,1,1};
    for (int d=rank-2; d>=0; --d) rstr[d] = rstr[d+1]*reg_dims[d+1];

    // grid-stride loop
    for (int64_t idx = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
        idx < total;
        idx += (int64_t)blockDim.x*gridDim.x)
    {
      int coord[4] = {0,0,0,0};
      #pragma unroll
      for (int d=0; d<4; ++d) {
        if (d<rank) coord[d] = (int)((idx / (int64_t)rstr[d]) % reg_dims[d]);
      }
      int64_t xoff = 0, yoff = 0;
      #pragma unroll
      for (int d=0; d<4; ++d){
        if (d<rank){
          xoff += (int64_t)(x_starts[d] + coord[d]) * x_strides[d];
          yoff += (int64_t)(y_starts[d] + coord[d]) * y_strides[d];
        }
      }
      Y[yoff] = X[xoff];
    }
  }

  template<int BS>
  __global__ void tensor_add_region_kernel(
      const float* __restrict__ S,
      float* __restrict__ D,
      int rank,
      const int* __restrict__ reg_dims,
      const int* __restrict__ s_strides,
      const int* __restrict__ d_strides,
      const int* __restrict__ s_starts,
      const int* __restrict__ d_starts)
  {
    int64_t total = 1;
    #pragma unroll
    for (int d=0; d<4; ++d) if (d<rank) total *= (int64_t)reg_dims[d];
    if (total <= 0) return;

    int rstr[4] = {1,1,1,1};
    for (int d=rank-2; d>=0; --d) rstr[d] = rstr[d+1]*reg_dims[d+1];

    for (int64_t idx = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
        idx < total;
        idx += (int64_t)blockDim.x*gridDim.x)
    {
      int coord[4] = {0,0,0,0};
      #pragma unroll
      for (int d=0; d<4; ++d) {
        if (d<rank) coord[d] = (int)((idx / (int64_t)rstr[d]) % reg_dims[d]);
      }
      int64_t soff = 0, doff = 0;
      #pragma unroll
      for (int d=0; d<4; ++d){
        if (d<rank){
          soff += (int64_t)(s_starts[d] + coord[d]) * s_strides[d];
          doff += (int64_t)(d_starts[d] + coord[d]) * d_strides[d];
        }
      }
      D[doff] += S[soff];
    }
  }
}

extern "C" void concat_copy_region_kernel_launcher(
  const float* X, float* Y, int rank,
  const int* reg_dims,
  const int* x_strides, const int* y_strides,
  const int* x_starts,  const int* y_starts,
  cudaStream_t s
){
  int64_t total = 1;
  for (int i=0;i<rank;++i) total *= reg_dims[i];
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1)/BS));
  tensor_copy_region_kernel<BS><<<grid, block, 0, s>>>(
    X,Y,rank,reg_dims,x_strides,y_strides,x_starts,y_starts
  );
}

extern "C" void concat_add_region_kernel_launcher(
  const float* S, float* D, int rank,
  const int* reg_dims,
  const int* s_strides, const int* d_strides,
  const int* s_starts,  const int* d_starts,
  cudaStream_t s
){
  int64_t total = 1;
  for (int i=0;i<rank;++i) total *= reg_dims[i];
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1)/BS));
  tensor_add_region_kernel<BS><<<grid, block, 0, s>>>(
    S,D,rank,reg_dims,s_strides,d_strides,s_starts,d_starts
  );
}
