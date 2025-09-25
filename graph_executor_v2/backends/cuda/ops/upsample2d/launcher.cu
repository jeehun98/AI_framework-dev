#include <cuda_runtime.h>
#include <stdint.h>
#include "backends/cuda/ops/upsample2d/api.hpp"

namespace ai {

static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

static inline bool is_nchw_f32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

// --- 좌표 매핑 유틸(Nearest) ---
// align_corners 규칙:
//  - true : ih = round( ho * (H-1) / max(Ho-1,1) )
//  - false: ih = floor( (ho + 0.5)*scale - 0.5 ), scale = H / Ho
// iw 역시 동일
__device__ __forceinline__
int map_nearest_index(int out_i, int out_len, int in_len, bool align_corners) {
  if (in_len == out_len) return min(out_i, in_len-1);
  if (align_corners) {
    if (out_len == 1) return 0;
    float pos = (float)out_i * (float)(in_len - 1) / (float)(out_len - 1);
    int idx = (int)roundf(pos);
    if (idx < 0) idx = 0; else if (idx >= in_len) idx = in_len - 1;
    return idx;
  } else {
    float scale = (float)in_len / (float)out_len;
    float pos = ((float)out_i + 0.5f) * scale - 0.5f;
    int idx = (int)floorf(pos);
    if (idx < 0) idx = 0; else if (idx >= in_len) idx = in_len - 1;
    return idx;
  }
}

// ============ FWD ============
__global__ void upsample2d_nearest_fwd_kernel(
  const float* __restrict__ X, float* __restrict__ Y,
  int N,int C,int H,int W, int Ho,int Wo, bool align_corners)
{
  const int nOut = N*C*Ho*Wo;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int ih = map_nearest_index(ho, Ho, H, align_corners);
  const int iw = map_nearest_index(wo, Wo, W, align_corners);

  const int in_base  = ((n*C + c)*H + ih)*W + iw;
  const int out_base = ((n*C + c)*Ho + ho)*Wo + wo;
  Y[out_base] = X[in_base];
}

// ============ BWD ============
// dX에 dY의 기여를 모아줌(여러 out이 같은 in에 매핑될 수 있으므로 atomic)
__global__ void upsample2d_nearest_bwd_kernel(
  const float* __restrict__ dY, float* __restrict__ dX,
  int N,int C,int H,int W, int Ho,int Wo, bool align_corners)
{
  const int nOut = N*C*Ho*Wo;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int ih = map_nearest_index(ho, Ho, H, align_corners);
  const int iw = map_nearest_index(wo, Wo, W, align_corners);

  const int out_base = ((n*C + c)*Ho + ho)*Wo + wo;
  const int in_base  = ((n*C + c)*H  + ih)*W  + iw;

  atomicAdd(&dX[in_base], dY[out_base]);
}

// --- 런처 ---
Status Upsample2DNearestCudaLaunch(const Tensor& X, Tensor& Y,
                                   const Upsample2DAttrs& attrs,
                                   StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return Status::Invalid;

  const int N=(int)X.desc.shape[0], C=(int)X.desc.shape[1];
  const int H=(int)X.desc.shape[2], W=(int)X.desc.shape[3];
  const int Ho=(int)Y.desc.shape[2], Wo=(int)Y.desc.shape[3];

  // 런치
  const int BS=256;
  const int nOut=N*C*Ho*Wo;
  dim3 block(BS), grid((nOut+BS-1)/BS);
  upsample2d_nearest_fwd_kernel<<<grid, block, 0, to_cuda(stream)>>>(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    N,C,H,W,Ho,Wo, attrs.align_corners
  );
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status Upsample2DNearestBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                           const Upsample2DAttrs& attrs,
                                           StreamHandle stream)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX)) return Status::Invalid;

  const int N=(int)dX.desc.shape[0], C=(int)dX.desc.shape[1];
  const int H=(int)dX.desc.shape[2], W=(int)dX.desc.shape[3];
  const int Ho=(int)dY.desc.shape[2], Wo=(int)dY.desc.shape[3];

  // dX=0 초기화
  cudaMemsetAsync(dX.data, 0, sizeof(float)*(size_t)N*C*H*W, to_cuda(stream));

  const int BS=256;
  const int nOut=N*C*Ho*Wo;
  dim3 block(BS), grid((nOut+BS-1)/BS);
  upsample2d_nearest_bwd_kernel<<<grid, block, 0, to_cuda(stream)>>>(
    static_cast<const float*>(dY.data),
    static_cast<float*>(dX.data),
    N,C,H,W,Ho,Wo, attrs.align_corners
  );
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
