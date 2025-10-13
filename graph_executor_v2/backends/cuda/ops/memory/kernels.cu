#include <cuda_runtime.h>
#include <cstdint>

namespace {

// grid-stride fill
template <typename T>
__global__ void fill_kernel(T* __restrict__ dst, int64_t N, T v) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int64_t i = tid; i < N; i += stride) {
    dst[i] = v;
  }
}

static inline dim3 pick_block(int64_t) { return dim3(256); }
static inline dim3 pick_grid (int64_t N) {
  int64_t blocks = (N + 256 - 1) / 256;
  if (blocks < 32)   blocks = 32;    // 캡처 친화: SM 조회 없이 보수적
  if (blocks > 4096) blocks = 4096;
  return dim3(static_cast<unsigned>(blocks));
}

} // anon

namespace ai {

void fill_scalar_f32_kernel_launcher(void* dst, int64_t N, float value, cudaStream_t s) {
  if (N <= 0) return;
  dim3 b = pick_block(N), g = pick_grid(N);
  fill_kernel<float><<<g, b, 0, s>>>(reinterpret_cast<float*>(dst), N, value);
}

void fill_scalar_i32_kernel_launcher(void* dst, int64_t N, int32_t value, cudaStream_t s) {
  if (N <= 0) return;
  dim3 b = pick_block(N), g = pick_grid(N);
  fill_kernel<int32_t><<<g, b, 0, s>>>(reinterpret_cast<int32_t*>(dst), N, value);
}

} // namespace ai
