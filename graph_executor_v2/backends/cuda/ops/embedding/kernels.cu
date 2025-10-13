// backends/cuda/ops/embedding/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace { // TU-local

__global__ void emb_forward_kernel(
  const float* __restrict__ W, int V, int D,
  const int* __restrict__ I, int N, int L,
  int padding_idx, float out_scale,
  float* __restrict__ Y, bool y_is_3d)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x; // over N*L
  int nl = N*L;
  if (t >= nl) return;

  int n = t / L;
  int l = t % L;
  int idx = I[t];

  if (idx < 0 || idx >= V || (padding_idx>=0 && idx==padding_idx)) {
    for (int d=threadIdx.y; d<D; d+=blockDim.y) {
      if (y_is_3d) Y[((n*L + l)*D) + d] = 0.f;
      else         Y[(l*D) + d] = 0.f;
    }
    return;
  }

  const float* wrow = W + ((size_t)idx * D);
  for (int d=threadIdx.y; d<D; d+=blockDim.y) {
    float v = wrow[d] * out_scale;
    if (y_is_3d) Y[((n*L + l)*D) + d] = v;
    else         Y[(l*D) + d] = v;
  }
}

__global__ void emb_backward_scatter_kernel(
  const int* __restrict__ I, int N, int L,
  const float* __restrict__ dY, bool dy_is_3d,
  int V, int D,
  int padding_idx,
  const int* __restrict__ freq, bool scale_grad_by_freq,
  float* __restrict__ dW)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x; // over N*L
  int nl = N*L;
  if (t >= nl) return;

  int n = t / L;
  int l = t % L;
  int idx = I[t];
  if (idx < 0 || idx >= V) return;
  if (padding_idx>=0 && idx==padding_idx) return;

  float scale = 1.f;
  if (scale_grad_by_freq && freq) {
    int f = freq[idx];
    if (f > 1) scale = 1.0f / (float)f;
  }

  const float* gyrow = dy_is_3d ? (dY + ((size_t)(n*L + l)*D)) : (dY + ((size_t)l*D));
  float* dwrow = dW + ((size_t)idx * D);

  for (int d=threadIdx.y; d<D; d+=blockDim.y) {
    float addv = gyrow[d] * scale;
    atomicAdd(&dwrow[d], addv);
  }
}

} // anonymous

namespace ai {

void embedding_forward_launcher(
  const float* W, int V, int D,
  const int* I, int N, int L,
  int padding_idx, float out_scale,
  float* Y, bool y_is_3d,
  cudaStream_t s)
{
  dim3 block(128, 4);
  dim3 grid( (N*L + block.x - 1)/block.x );
  emb_forward_kernel<<<grid, block, 0, s>>>(
    W, V, D, I, N, L, padding_idx, out_scale, Y, y_is_3d);
}

void embedding_backward_scatter_launcher(
  const int* I, int N, int L,
  const float* dY, bool dy_is_3d,
  int V, int D,
  int padding_idx,
  const int* freq, bool scale_grad_by_freq,
  float* dW,
  cudaStream_t s)
{
  dim3 block(128, 4);
  dim3 grid( (N*L + block.x - 1)/block.x );
  emb_backward_scatter_kernel<<<grid, block, 0, s>>>(
    I, N, L, dY, dy_is_3d, V, D, padding_idx, freq, scale_grad_by_freq, dW);
}

void count_frequency_launcher(
  const int* I, int N, int L, int V,
  int* out_freq, cudaStream_t s)
{
  // (옵션) 필요 시 구현
  (void)I; (void)N; (void)L; (void)V; (void)out_freq; (void)s;
}

} // namespace ai
