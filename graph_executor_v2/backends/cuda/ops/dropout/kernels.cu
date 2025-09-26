// kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ inline uint64_t splitmix64(uint64_t x){
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ inline float u01_from_u64(uint64_t s){
  const uint32_t r = static_cast<uint32_t>(s >> 40); // 상위 24비트
  return (r + 0.5f) / 16777216.0f; // [0,1)
}

template<int BS>
__global__ void dropout_fwd_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int32_t* __restrict__ M,  // may be null
                                   int Mrows, int Ncols,
                                   float p, float scale,
                                   uint64_t seed,
                                   uint64_t counter_base)     // NEW
{
  const int row = blockIdx.x;
  if (row >= Mrows) return;
  const int stride = BS;
  const int base = row * Ncols;

  for (int j = threadIdx.x; j < Ncols; j += stride) {
    const int idx = base + j;
    // stateless RNG: seed ^ (counter_base + idx)
    const uint64_t gidx = counter_base + static_cast<uint64_t>(idx);
    const float u = u01_from_u64(splitmix64(seed ^ gidx));

    const int32_t m = (u >= p) ? 1 : 0;
    float y = static_cast<float>(m) * X[idx];
    if (scale) y *= scale;   // scale_in_train ? 1/(1-p) : 1
    Y[idx] = y;
    if (M) M[idx] = m;
  }
}

template<int BS>
__global__ void dropout_bwd_kernel(const float* __restrict__ dY,
                                   const int32_t* __restrict__ M,
                                   float* __restrict__ dX,
                                   int Mrows, int Ncols,
                                   float scale)
{
  const int row = blockIdx.x;
  if (row >= Mrows) return;
  const int stride = BS;
  const int base = row * Ncols;

  for (int j = threadIdx.x; j < Ncols; j += stride) {
    const int idx = base + j;
    dX[idx] = static_cast<float>(M[idx]) * dY[idx] * scale;
  }
}

} // anonymous

namespace ai {

// Forward launcher: counter_base 파라미터 추가 (필수)
void dropout_forward_kernel_launcher(const float* X,
                                     float* Y,
                                     int32_t* mask,
                                     int M_rows,
                                     int N_cols,
                                     float p,
                                     bool scale_in_train,
                                     uint64_t seed,
                                     uint64_t counter_base,   // NEW
                                     cudaStream_t s)
{
  constexpr int BS = 256;
  const float scale = scale_in_train ? ((p < 1.f) ? (1.f / (1.f - p)) : 0.f) : 1.f;
  dim3 grid(M_rows), block(BS);
  dropout_fwd_kernel<BS><<<grid, block, 0, s>>>(
      X, Y, mask, M_rows, N_cols, p, scale, seed, counter_base);
}

void dropout_backward_kernel_launcher(const float* dY,
                                      const int32_t* mask,
                                      float* dX,
                                      int M_rows,
                                      int N_cols,
                                      float p,
                                      bool scale_in_train,
                                      cudaStream_t s)
{
  constexpr int BS = 256;
  const float scale = scale_in_train ? ((p < 1.f) ? (1.f / (1.f - p)) : 0.f) : 1.f;
  dim3 grid(M_rows), block(BS);
  dropout_bwd_kernel<BS><<<grid, block, 0, s>>>(
      dY, mask, dX, M_rows, N_cols, scale);
}

} // namespace ai
