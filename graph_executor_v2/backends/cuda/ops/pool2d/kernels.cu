// backends/cuda/ops/pool2d/kernels.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>
#include <math_constants.h>
#include <algorithm>   // max, min
#include <limits>

// ====== 공용 호스트 유틸 ======
namespace {
inline int div_up_host(int a, int b) { return (a + b - 1) / b; }

inline void output_dims_2d_host(
  int H, int W, int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,bool ceil_mode,
  int& Ho, int& Wo)
{
  const int effKH = (kH - 1) * dH + 1;
  const int effKW = (kW - 1) * dW + 1;
  const int aH = H + 2 * pH - effKH;
  const int aW = W + 2 * pW - effKW;

  if (ceil_mode) {
    Ho = (aH >= 0 ? div_up_host(aH, sH) + 1 : 0);
    Wo = (aW >= 0 ? div_up_host(aW, sW) + 1 : 0);
  } else {
    Ho = (aH >= 0 ? (aH / sH) + 1 : 0);
    Wo = (aW >= 0 ? (aW / sW) + 1 : 0);
  }
  if (Ho < 0) Ho = 0;
  if (Wo < 0) Wo = 0;
}
} // anonymous

// ============ MaxPool2D ============
// NOTE(타이브레이크): 동일 값이면 '첫 번째'를 유지합니다.
// NOTE(NaN): 입력에 NaN이 섞인 경우 그 위치를 선택하려면 isnan 분기를 켜세요.
__global__ void maxpool2d_fwd_kernel(
  const float* __restrict__ X, float* __restrict__ Y, int32_t* __restrict__ Ind,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode,
  int Ho, int Wo)
{
  const long long nOut = 1LL * N * C * Ho * Wo;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nOut;
       tid += 1LL * blockDim.x * gridDim.x)
  {
    long long tmp = tid;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int c  = tmp % C;  tmp /= C;
    const int n  = (int)tmp;

    const int base = ((n*C + c)*H)*W;
    const int hstart = ho * sH - pH;
    const int wstart = wo * sW - pW;

    float best = -CUDART_INF_F;
    int   best_idx = 0;

    for (int kh=0; kh<kH; ++kh) {
      const int ih = hstart + kh * dH;
      if (ih < 0 || ih >= H) continue;
      const int row = base + ih*W;
      for (int kw=0; kw<kW; ++kw) {
        const int iw = wstart + kw * dW;
        if (iw < 0 || iw >= W) continue;
        const int idx = row + iw;
        const float v = X[idx];

        // NaN 우선 선택을 원할 경우 아래 분기 활성화:
        // if (isnan(v)) { best = v; best_idx = ih*W + iw; goto write_out; }

        if (v > best) {
          best = v;
          best_idx = ih*W + iw;
        }
      }
    }

    const int out_idx = ((n*C + c)*Ho + ho)*Wo + wo;
    Y[out_idx] = best;
    if (Ind) Ind[out_idx] = best_idx;
  }
}

__global__ void maxpool2d_bwd_kernel(
  const float* __restrict__ dY, const int32_t* __restrict__ Ind, float* __restrict__ dX,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode,
  int Ho, int Wo)
{
  const long long nOut = 1LL * N * C * Ho * Wo;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nOut;
       tid += 1LL * blockDim.x * gridDim.x)
  {
    long long tmp = tid;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int c  = tmp % C;  tmp /= C;
    const int n  = (int)tmp;

    const int out_idx = ((n*C + c)*Ho + ho)*Wo + wo;
    const float g = dY[out_idx];
    const int32_t best = Ind[out_idx];

    // best는 ih*W + iw
    const int ih = best / W;
    const int iw = best % W;

    // 혹시라도 잘못된 인덱스가 들어오는 경우 방어(디버그 빌드에서만 써도 좋음)
    if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
      const int in_idx = ((n*C + c)*H + ih)*W + iw;
      atomicAdd(&dX[in_idx], g);
    }
  }
}

// ============ AvgPool2D ============
__global__ void avgpool2d_fwd_kernel(
  const float* __restrict__ X, float* __restrict__ Y,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode, bool count_include_pad,
  int Ho, int Wo)
{
  const long long nOut = 1LL * N * C * Ho * Wo;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nOut;
       tid += 1LL * blockDim.x * gridDim.x)
  {
    long long tmp = tid;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int c  = tmp % C;  tmp /= C;
    const int n  = (int)tmp;

    const int base = ((n*C + c)*H)*W;
    const int hstart = ho * sH - pH;
    const int wstart = wo * sW - pW;

    float sum = 0.f;
    int cnt = 0;

    for (int kh=0; kh<kH; ++kh) {
      const int ih = hstart + kh * dH;
      for (int kw=0; kw<kW; ++kw) {
        const int iw = wstart + kw * dW;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
          sum += X[base + ih*W + iw];
          ++cnt;
        } else if (count_include_pad) {
          ++cnt;
        }
      }
    }
    if (cnt == 0) cnt = 1;

    const int out_idx = ((n*C + c)*Ho + ho)*Wo + wo;
    Y[out_idx] = sum / (float)cnt;
  }
}

__global__ void avgpool2d_bwd_kernel(
  const float* __restrict__ dY, float* __restrict__ dX,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode, bool count_include_pad,
  int Ho, int Wo)
{
  const long long nOut = 1LL * N * C * Ho * Wo;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nOut;
       tid += 1LL * blockDim.x * gridDim.x)
  {
    long long tmp = tid;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int c  = tmp % C;  tmp /= C;
    const int n  = (int)tmp;

    const int base = ((n*C + c)*H)*W;
    const int hstart = ho * sH - pH;
    const int wstart = wo * sW - pW;

    int cnt = 0;
    for (int kh=0; kh<kH; ++kh) {
      const int ih = hstart + kh * dH;
      for (int kw=0; kw<kW; ++kw) {
        const int iw = wstart + kw * dW;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) ++cnt;
        else if (count_include_pad) ++cnt;
      }
    }
    if (cnt == 0) cnt = 1;

    const float g = dY[((n*C + c)*Ho + ho)*Wo + wo] / (float)cnt;

    for (int kh=0; kh<kH; ++kh) {
      const int ih = hstart + kh * dH;
      if ((unsigned)ih >= (unsigned)H) continue;
      for (int kw=0; kw<kW; ++kw) {
        const int iw = wstart + kw * dW;
        if ((unsigned)iw >= (unsigned)W) continue;
        atomicAdd(&dX[base + ih*W + iw], g);
      }
    }
  }
}

// ====== 런처 ======
namespace ai {

// occupancy 기반 런처 헬퍼 (커널 포인터 버전)
template <typename KernelPtr>
static inline void launch_config_for(int nOut, int& grid, int& block,
                                     KernelPtr kptr,
                                     size_t dyn_smem_bytes = 0) {
  int minGrid = 0, blockSize = 0;
  // kptr: __global__ 커널의 함수 포인터여야 합니다.
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kptr,
                                     dyn_smem_bytes, /*blockSizeLimit=*/0);

  block = (blockSize > 0 ? blockSize : 256);
  grid  = (nOut + block - 1) / block;
  grid  = std::min(grid, 65535); // 안전 가드
}

void maxpool2d_fwd_kernel_launcher(const float* X, float* Y, int32_t* Ind,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const long long nOut = 1LL * N * C * Ho * Wo;

  int minGrid=0, blockSize=0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
    maxpool2d_fwd_kernel, 0, 0);
  if (blockSize <= 0) blockSize = 256;
  int grid = (int)std::min( (nOut + blockSize - 1) / blockSize, 1LL * 65535 );

  maxpool2d_fwd_kernel<<<grid, blockSize, 0, s>>>(
      X, Y, Ind, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode, Ho, Wo);
}

void maxpool2d_bwd_kernel_launcher(const float* dY, const int32_t* Ind, float* dX,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, cudaStream_t s)
{
  int Ho, Wo;
  output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode, Ho, Wo);
  const long long nOut = 1LL * N * C * Ho * Wo;

  int minGrid=0, blockSize=0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
    maxpool2d_bwd_kernel, 0, 0);
  if (blockSize <= 0) blockSize = 256;
  int grid = (int)std::min( (nOut + blockSize - 1) / blockSize, 1LL * 65535 );

  maxpool2d_bwd_kernel<<<grid, blockSize, 0, s>>>(
      dY, Ind, dX, N,C,H,W, kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode, Ho, Wo);
}

void avgpool2d_fwd_kernel_launcher(const float* X, float* Y,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, bool count_include_pad, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const long long nOut = 1LL * N * C * Ho * Wo;

  int minGrid=0, blockSize=0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
    avgpool2d_fwd_kernel, 0, 0);
  if (blockSize <= 0) blockSize = 256;
  int grid = (int)std::min( (nOut + blockSize - 1) / blockSize, 1LL * 65535 );

  avgpool2d_fwd_kernel<<<grid, blockSize, 0, s>>>(
      X, Y, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,count_include_pad, Ho, Wo);
}

void avgpool2d_bwd_kernel_launcher(const float* dY, float* dX,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, bool count_include_pad, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const long long nOut = 1LL * N * C * Ho * Wo;

  int minGrid=0, blockSize=0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
    avgpool2d_bwd_kernel, 0, 0);
  if (blockSize <= 0) blockSize = 256;
  int grid = (int)std::min( (nOut + blockSize - 1) / blockSize, 1LL * 65535 );

  avgpool2d_bwd_kernel<<<grid, blockSize, 0, s>>>(
      dY, dX, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,count_include_pad, Ho, Wo);
}

} // namespace ai
