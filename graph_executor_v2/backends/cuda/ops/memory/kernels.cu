#include <cuda_runtime.h>
#include <stdint.h>
#include "ai/tensor.hpp"

// 최대 차원 상한 (필요시 늘려도 됨)
#ifndef MEM_MAX_NDIMS
#define MEM_MAX_NDIMS 8
#endif

// grid-stride loop용
static __device__ __forceinline__ int64_t grid_stride_loop_start() {
  return blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
}
static __device__ __forceinline__ int64_t grid_stride_loop_stride() {
  return (int64_t)gridDim.x * blockDim.x;
}

// ND → row-major(연속) 복사 커널
__global__ void contiguous_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ stride,   // 요소 단위
    const int64_t* __restrict__ pitch,    // row-major 분해용 (제품 누적)
    int nd,
    int64_t total)
{
  int64_t i = grid_stride_loop_start();
  int64_t step = grid_stride_loop_stride();

  for (; i < total; i += step) {
    // i 를 row-major 기준의 ND 인덱스로 분해
    int64_t rem = i;
    int64_t off = 0;
#pragma unroll
    for (int d = 0; d < MEM_MAX_NDIMS; ++d) {
      if (d >= nd) break;
      const int64_t pd = pitch[d];            // prod(shape[d+1..])
      const int64_t idx_d = (pd == 0) ? 0 : (rem / pd);
      rem -= idx_d * pd;
      off += idx_d * stride[d];
    }
    dst[i] = src[off];
  }
}

// ---- 런처 시그니처 ----
namespace ai {
void contiguous_copy_kernel_launcher(
    const float* src, float* dst,
    const int64_t* shape_h, const int64_t* stride_h,
    int nd, int64_t total, cudaStream_t stream)
{
  // pitch 계산 (host)
  int64_t pitch_h[MEM_MAX_NDIMS] = {0};
  {
    int64_t acc = 1;
    for (int d = nd - 1; d >= 0; --d) {
      pitch_h[d] = acc;
      acc *= shape_h[d];
    }
  }

  // 장치 스택/상수 대신 소량 버퍼로 복사
  int64_t *d_shape=nullptr, *d_stride=nullptr, *d_pitch=nullptr;
  cudaMalloc(&d_shape,  sizeof(int64_t) * nd);
  cudaMalloc(&d_stride, sizeof(int64_t) * nd);
  cudaMalloc(&d_pitch,  sizeof(int64_t) * nd);
  cudaMemcpyAsync(d_shape,  shape_h,  sizeof(int64_t)*nd, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_stride, stride_h, sizeof(int64_t)*nd, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_pitch,  pitch_h,  sizeof(int64_t)*nd, cudaMemcpyHostToDevice, stream);

  // 그리드
  const int BS = 256;
  const int GS = (int)((total + BS - 1) / BS);
  contiguous_copy_kernel<<<GS, BS, 0, stream>>>(src, dst, d_shape, d_stride, d_pitch, nd, total);

  // 정리
  cudaFree(d_shape);
  cudaFree(d_stride);
  cudaFree(d_pitch);
}
} // namespace ai
