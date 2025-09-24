#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// (outer, K, inner)로 X를 보고, (outer, M, inner)로 Y/Index를 본다.
// 인덱스는 [0, K) 범위여야 한다.
__global__ void gather_axis_kernel(
    const float* __restrict__ X, const int32_t* __restrict__ Index, float* __restrict__ Y,
    int outer, int K, int inner, int M)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = outer * M * inner;
  if (tid >= n) return;

  int tmp = tid;
  int i = tmp % inner; tmp /= inner;   // inner
  int m = tmp % M;     tmp /= M;       // M (output axis)
  int o = tmp;                          // outer

  int idx = Index[( (o * M) + m ) * inner + i];
  if ((unsigned)idx >= (unsigned)K) {
    // out-of-range 보호: 무시(0) 또는 클램프. 여기선 무시.
    return;
  }

  // 선형 인덱싱 (X: outer x K x inner), (Y: outer x M x inner)
  int xoff = ((o * K) + idx) * inner + i;
  int yoff = ((o * M) +  m ) * inner + i;
  Y[yoff] = X[xoff];
}

__global__ void scatter_add_axis_kernel(
    float* __restrict__ Out, const int32_t* __restrict__ Index, const float* __restrict__ Src,
    int outer, int K, int inner, int M)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = outer * M * inner;
  if (tid >= n) return;

  int tmp = tid;
  int i = tmp % inner; tmp /= inner;   // inner
  int m = tmp % M;     tmp /= M;       // M
  int o = tmp;                          // outer

  int idx = Index[( (o * M) + m ) * inner + i];
  if ((unsigned)idx >= (unsigned)K) {
    // out-of-range 인덱스는 무시
    return;
  }

  int off = ((o * K) + idx) * inner + i;
  float val = Src[( (o * M) + m ) * inner + i];
  atomicAdd(&Out[off], val);
}

} // anon

namespace ai {

// 런처(내부 전용) — axis를 outer/K/inner, M으로 변환해 커널 호출
void gather_axis_launch(const float* X, const int32_t* Index, float* Y,
                        int outer, int K, int inner, int M, cudaStream_t s)
{
  int BS = 256;
  int n  = outer * M * inner;
  dim3 grid((n + BS - 1)/BS), block(BS);
  gather_axis_kernel<<<grid, block, 0, s>>>(X, Index, Y, outer, K, inner, M);
}

void scatter_add_axis_launch(float* Out, const int32_t* Index, const float* Src,
                             int outer, int K, int inner, int M, cudaStream_t s)
{
  int BS = 256;
  int n  = outer * M * inner;
  dim3 grid((n + BS - 1)/BS), block(BS);
  scatter_add_axis_kernel<<<grid, block, 0, s>>>(Out, Index, Src, outer, K, inner, M);
}

} // namespace ai
