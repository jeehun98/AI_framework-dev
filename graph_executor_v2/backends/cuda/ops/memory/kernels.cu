#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ===== 내부 커널 =====
namespace {

template<typename T>
__global__ void kfill(T* __restrict__ dst, T v, size_t n)
{
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += (size_t)blockDim.x * gridDim.x)
  {
    dst[i] = v;
  }
}

} // anonymous

// ===== 외부로 노출되는 C-링크 런처 (host 함수) =====
extern "C" void fill_f32_kernel_launcher(
    uint64_t dst_ptr,
    size_t n,
    float value,
    cudaStream_t s)
{
  if (n == 0) return;
  constexpr int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  grid = grid > 0 ? grid : 1;
  grid = grid > 65535 ? 65535 : grid;

  auto* p = reinterpret_cast<float*>(dst_ptr);
  kfill<float><<<grid, BS, 0, s>>>(p, value, n);
}

extern "C" void fill_i32_kernel_launcher(
    uint64_t dst_ptr,
    size_t n,
    int32_t value,
    cudaStream_t s)
{
  if (n == 0) return;
  constexpr int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  grid = grid > 0 ? grid : 1;
  grid = grid > 65535 ? 65535 : grid;

  auto* p = reinterpret_cast<int32_t*>(dst_ptr);
  kfill<int32_t><<<grid, BS, 0, s>>>(p, value, n);
}
