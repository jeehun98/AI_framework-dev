#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ===== 내부 커널/유틸 =====
namespace {

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z = z ^ (z >> 31);
  return z;
}

__device__ __forceinline__ float u01_from_u32(uint32_t u) {
  return (float)(u) * (1.0f / 4294967296.0f); // 2^-32
}

__device__ __forceinline__ float rand_uniform(uint64_t seed, uint64_t ctr) {
  uint64_t v = splitmix64(seed ^ ctr);
  uint32_t u = (uint32_t)(v & 0xFFFFFFFFu);
  return u01_from_u32(u);
}

// grid-stride loop로 큰 n 안전 처리
__global__ void dropout_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int32_t* __restrict__ mask, // nullable
                                       size_t n,
                                       float p,
                                       bool scale_in_train,
                                       uint64_t seed,
                                       uint64_t counter_base)
{
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += (size_t)blockDim.x * gridDim.x)
  {
    float r = rand_uniform(seed, counter_base + i); // stateless
    int m = (r >= p) ? 1 : 0;                        // keep prob = 1-p

    float v = x[i] * (float)m;
    if (scale_in_train && (1.0f - p) > 0.f) {
      v *= (1.0f / (1.0f - p));
    }
    y[i] = v;
    if (mask) mask[i] = m;
  }
}

__global__ void dropout_backward_kernel(const float* __restrict__ gy,
                                        const int32_t* __restrict__ mask,
                                        float* __restrict__ gx,
                                        size_t n,
                                        float p,
                                        bool scale_in_train)
{
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += (size_t)blockDim.x * gridDim.x)
  {
    int m = mask[i];
    float v = gy[i] * (float)m;
    if (scale_in_train && (1.0f - p) > 0.f) {
      v *= (1.0f / (1.0f - p));
    }
    gx[i] = v;
  }
}

} // anonymous

// ===== 외부로 노출되는 C-링크 런처 (host 함수) =====
extern "C" void dropout_forward_kernel_launcher(
    const float* x, float* y, int32_t* mask,
    size_t n,
    float p, bool scale_in_train,
    uint64_t seed, uint64_t counter_base,
    cudaStream_t s)
{
  if (n == 0) return;
  constexpr int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  grid = grid > 0 ? grid : 1;
  grid = grid > 65535 ? 65535 : grid; // 보수적 상한

  dropout_forward_kernel<<<grid, BS, 0, s>>>(
      x, y, mask, n, p, scale_in_train, seed, counter_base
  );
}

extern "C" void dropout_backward_kernel_launcher(
    const float* gy, const int32_t* mask, float* gx,
    size_t n,
    float p, bool scale_in_train,
    cudaStream_t s)
{
  if (n == 0) return;
  constexpr int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  grid = grid > 0 ? grid : 1;
  grid = grid > 65535 ? 65535 : grid;

  dropout_backward_kernel<<<grid, BS, 0, s>>>(
      gy, mask, gx, n, p, scale_in_train
  );
}
