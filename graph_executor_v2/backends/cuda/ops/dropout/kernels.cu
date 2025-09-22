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
  // 24bit mantissa로 float 변환
  const uint32_t r = static_cast<uint32_t>(s >> 40); // 상위 24비트
  return (r + 0.5f) / 16777216.0f; // [0,1)
}

template<int BS>
__global__ void dropout_fwd_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int32_t* __restrict__ M, // may be null
                                   int Mrows, int Ncols,
                                   float p, float scale, uint64_t seed)
{
  const int row = blockIdx.x;
  if (row >= Mrows) return;
  const int stride = BS;
  const int base = row * Ncols;

  for (int j = threadIdx.x; j < Ncols; j += stride) {
    const int idx = base + j;
    // stateless RNG: (seed ^ idx) -> splitmix64 -> uniform
    float u = u01_from_u64(splitmix64(seed ^ static_cast<uint64_t>(idx)));
    int32_t m = (u >= p) ? 1 : 0;
    float y = static_cast<float>(m) * X[idx];
    if (scale) y *= scale;
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

// kernels.cu 내부 (namespace ai 안)

void dropout_forward_kernel_launcher(const float* X,
                                     float* Y,
                                     int32_t* mask,      // ← 이름 변경
                                     int M_rows,         // ← 이름 변경
                                     int N_cols,         // ← 이름 변경
                                     float p,
                                     bool scale_in_train,
                                     uint64_t seed,
                                     cudaStream_t s)
{
  constexpr int BS = 256;
  const float scale = scale_in_train ? ((p < 1.f) ? (1.f / (1.f - p)) : 0.f) : 1.f;
  dim3 grid(M_rows), block(BS);
  dropout_fwd_kernel<BS><<<grid, block, 0, s>>>(
      X, Y, mask, M_rows, N_cols, p, scale, seed);
}

void dropout_backward_kernel_launcher(const float* dY,
                                      const int32_t* mask, // ← 이름 변경
                                      float* dX,
                                      int M_rows,          // ← 이름 변경
                                      int N_cols,          // ← 이름 변경
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
