// backends/cuda/ops/dropout/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

// Stateless RNG: splitmix64 -> xorshift32 변형으로 U(0,1)
static __device__ __forceinline__ uint64_t splitmix64(uint64_t x){
  x += 0x9e3779b97f4a7c15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
static __device__ __forceinline__ float u01_from_u64(uint64_t u){
  // take high 23 bits to mantissa (0,1)
  const uint32_t mant = (uint32_t)((u >> 41) | 1); // avoid exact 0
  // scale to (0,1): mant / 2^23
  return (float)mant * (1.0f / 8388608.0f);
}

template<int BS>
__global__ void dropout_fwd_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int32_t* __restrict__ Mask,
                                   int M, int N,
                                   float p, bool scale_in_train,
                                   uint64_t seed, uint64_t counter_base)
{
  const int row = blockIdx.x;
  if (row >= M) return;

  const int64_t base = (int64_t)row * N;
  const float scale = (scale_in_train && p < 1.0f) ? (1.0f / (1.0f - p)) : 1.0f;

  for (int col = threadIdx.x; col < N; col += BS) {
    const int64_t idx = base + col;

    // Stateless RNG: per-element counter = counter_base + idx
    const uint64_t ctr = counter_base + (uint64_t)idx;
    const uint64_t rnd = splitmix64(seed ^ ctr);
    const float r = u01_from_u64(rnd); // (0,1)

    const bool keep = (r >= p);
    const float x = X[idx];
    const float y = keep ? (x * scale) : 0.0f;

    Y[idx] = y;
    if (Mask) Mask[idx] = keep ? 1 : 0;
  }
}

template<int BS>
__global__ void dropout_bwd_kernel(const float* __restrict__ dY,
                                   const int32_t* __restrict__ Mask,
                                   float* __restrict__ dX,
                                   int M, int N,
                                   float p, bool scale_in_train)
{
  (void)p; // not needed
  const int row = blockIdx.x;
  if (row >= M) return;

  const int64_t base = (int64_t)row * N;
  const float scale = (scale_in_train && p < 1.0f) ? (1.0f / (1.0f - p)) : 1.0f;

  for (int col = threadIdx.x; col < N; col += BS) {
    const int64_t idx = base + col;
    dX[idx] = (Mask[idx] != 0) ? (dY[idx] * scale) : 0.0f;
  }
}

namespace ai {

void dropout_fwd_kernel_launcher(const float* x, float* y, int32_t* mask,
                                 int M, int N,
                                 float p, bool scale_in_train,
                                 uint64_t seed, uint64_t counter_base,
                                 cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  dropout_fwd_kernel<BS><<<grid, block, 0, s>>>(
      x, y, mask, M, N, p, scale_in_train, seed, counter_base);
}

void dropout_bwd_kernel_launcher(const float* dy, const int32_t* mask, float* dx,
                                 int M, int N,
                                 float p, bool scale_in_train,
                                 cudaStream_t s)
{
  constexpr int BS = 256;
  dim3 grid(M), block(BS);
  dropout_bwd_kernel<BS><<<grid, block, 0, s>>>(
      dy, mask, dx, M, N, p, scale_in_train);
}

} // namespace ai
