#include <cuda_runtime.h>
#include <cstdint>

namespace {

// ======================= Forward =======================
template<int BS>
__global__ void slice_forward_kernel(const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     int rank,
                                     const int* __restrict__ x_dims,    // [4]
                                     const int* __restrict__ x_strides, // [4]
                                     const int* __restrict__ y_dims,    // [4]
                                     const int* __restrict__ y_strides, // [4] row-major
                                     const int* __restrict__ starts)    // [4]
{
  // 총 원소 수: ∏(y_dims[0..rank-1])
  int64_t total = 1;
  #pragma unroll
  for (int i=0;i<4;++i) {
    if (i < rank) total *= (int64_t)y_dims[i];
  }

  int64_t idx = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= total) return;

  // y-linear -> multi-index (row-major)
  int coord[4] = {0,0,0,0};
  #pragma unroll
  for (int d=0; d<4; ++d) {
    if (d < rank) {
      coord[d] = (int)((idx / (int64_t)y_strides[d]) % y_dims[d]);
    }
  }

  // src offset = Σ (coord[d]+starts[d]) * x_strides[d]
  int64_t xoff = 0;
  #pragma unroll
  for (int d=0; d<4; ++d) {
    if (d < rank) {
      int xi = coord[d] + starts[d];
      xoff += (int64_t)xi * x_strides[d];
    }
  }

  // dst offset = idx (row-major)
  Y[idx] = X[xoff];
}

// ======================= Backward (accumulate) =======================
template<int BS>
__global__ void slice_backward_kernel(const float* __restrict__ gY,
                                      float* __restrict__ gX,
                                      int rank,
                                      const int* __restrict__ x_dims,
                                      const int* __restrict__ x_strides,
                                      const int* __restrict__ y_dims,
                                      const int* __restrict__ y_strides,
                                      const int* __restrict__ starts)
{
  int64_t total = 1;
  #pragma unroll
  for (int i=0;i<4;++i) {
    if (i < rank) total *= (int64_t)y_dims[i];
  }

  int64_t idx = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= total) return;

  int coord[4] = {0,0,0,0};
  #pragma unroll
  for (int d=0; d<4; ++d) {
    if (d < rank) {
      coord[d] = (int)((idx / (int64_t)y_strides[d]) % y_dims[d]);
    }
  }

  int64_t xoff = 0;
  #pragma unroll
  for (int d=0; d<4; ++d) {
    if (d < rank) {
      int xi = coord[d] + starts[d];
      xoff += (int64_t)xi * x_strides[d];
    }
  }

  gX[xoff] += gY[idx];
}

} // anonymous namespace


// ===== Host-callable launchers (kernels only see cudaStream_t) =====
extern "C" void slice_forward_kernel_launcher(const float* X, float* Y,
                                              int rank,
                                              const int* x_dims,
                                              const int* x_strides,
                                              const int* y_dims,
                                              const int* y_strides,
                                              const int* starts,
                                              cudaStream_t s)
{
  int64_t total = 1;
  for (int i=0;i<rank;++i) total *= y_dims[i];
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1) / BS));
  slice_forward_kernel<BS><<<grid, block, 0, s>>>(X, Y, rank, x_dims, x_strides, y_dims, y_strides, starts);
}

extern "C" void slice_backward_kernel_launcher(const float* gY, float* gX,
                                               int rank,
                                               const int* x_dims,
                                               const int* x_strides,
                                               const int* y_dims,
                                               const int* y_strides,
                                               const int* starts,
                                               cudaStream_t s)
{
  int64_t total = 1;
  for (int i=0;i<rank;++i) total *= y_dims[i];
  constexpr int BS = 256;
  dim3 block(BS), grid((int)((total + BS - 1) / BS));
  slice_backward_kernel<BS><<<grid, block, 0, s>>>(gY, gX, rank, x_dims, x_strides, y_dims, y_strides, starts);
}
