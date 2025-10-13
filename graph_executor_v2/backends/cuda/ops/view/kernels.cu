#include <cuda_runtime.h>
#include <cstdint>

namespace {

template<int BS>
__global__ void tensor_copy_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int64_t total){
  int64_t idx = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
  if (idx < total) Y[idx] = X[idx];
}

template<int BS>
__global__ void tensor_add_kernel(const float* __restrict__ S,
                                  float* __restrict__ D,
                                  int64_t total){
  int64_t idx = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
  if (idx < total) D[idx] += S[idx];
}

} // anon

extern "C" void view_copy_kernel_launcher(const float* X, float* Y, int64_t total, cudaStream_t s){
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1)/BS));
  tensor_copy_kernel<BS><<<grid, block, 0, s>>>(X, Y, total);
}

extern "C" void view_add_kernel_launcher(const float* S, float* D, int64_t total, cudaStream_t s){
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1)/BS));
  tensor_add_kernel<BS><<<grid, block, 0, s>>>(S, D, total);
}
