// backends/cuda/ops/pool2d/kernels.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>
#include <math_constants.h>

// ====== 공용 호스트 유틸 ======
namespace {
// 정식 공식: floor/ceil((in + 2p - effK) / s) + 1
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
__global__ void maxpool2d_fwd_kernel(
  const float* __restrict__ X, float* __restrict__ Y, int32_t* __restrict__ Ind,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode,
  int Ho, int Wo)                                  // <<< 추가
{
  const int nOut = N*C*Ho*Wo;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int base = ((n*C + c)*H)*W;

  const int hstart = ho * sH - pH;
  const int wstart = wo * sW - pW;

  float best = -CUDART_INF_F;
  int best_idx = 0;

  for (int kh=0; kh<kH; ++kh) {
    const int ih = hstart + kh * dH;
    if (ih < 0 || ih >= H) continue;
    for (int kw=0; kw<kW; ++kw) {
      const int iw = wstart + kw * dW;
      if (iw < 0 || iw >= W) continue;
      const int idx = base + ih*W + iw;
      const float v = X[idx];
      if (v > best) { best = v; best_idx = ih*W + iw; }
    }
  }

  const int out_idx = ((n*C + c)*Ho + ho)*Wo + wo;
  Y[out_idx] = best;
  if (Ind) Ind[out_idx] = best_idx;
}

__global__ void maxpool2d_bwd_kernel(
  const float* __restrict__ dY, const int32_t* __restrict__ Ind, float* __restrict__ dX,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode,
  int Ho, int Wo)                                  // <<< 추가
{
  const int nOut = N*C*Ho*Wo;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int out_idx = ((n*C + c)*Ho + ho)*Wo + wo;
  const float g = dY[out_idx];
  const int32_t best = Ind[out_idx];
  const int ih = best / W;
  const int iw = best % W;
  const int in_idx = ((n*C + c)*H + ih)*W + iw;

  atomicAdd(&dX[in_idx], g);
}

// ============ AvgPool2D ============
__global__ void avgpool2d_fwd_kernel(
  const float* __restrict__ X, float* __restrict__ Y,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode, bool count_include_pad,
  int Ho, int Wo)                                  // <<< 추가
{
  const int nOut = N*C*Ho*Wo;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int base = ((n*C + c)*H)*W;
  const int hstart = ho * sH - pH;
  const int wstart = wo * sW - pW;

  float sum = 0.f;
  int cnt = 0;
  for (int kh=0; kh<kH; ++kh) {
    const int ih = hstart + kh * dH;
    for (int kw=0; kw<kW; ++kw) {
      const int iw = wstart + kw * dW;
      if (ih >=0 && ih < H && iw >=0 && iw < W) {
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

__global__ void avgpool2d_bwd_kernel(
  const float* __restrict__ dY, float* __restrict__ dX,
  int N,int C,int H,int W,
  int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
  bool ceil_mode, bool count_include_pad,
  int Ho, int Wo)                                  // <<< 추가
{
  const int nOut = N*C*Ho*Wo;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nOut) return;

  int tmp = tid;
  const int wo = tmp % Wo; tmp /= Wo;
  const int ho = tmp % Ho; tmp /= Ho;
  const int c  = tmp % C;  tmp /= C;
  const int n  = tmp;

  const int base = ((n*C + c)*H)*W;
  const int hstart = ho * sH - pH;
  const int wstart = wo * sW - pW;

  int cnt = 0;
  for (int kh=0; kh<kH; ++kh) {
    const int ih = hstart + kh * dH;
    for (int kw=0; kw<kW; ++kw) {
      const int iw = wstart + kw * dW;
      if (ih >=0 && ih < H && iw >=0 && iw < W) ++cnt;
      else if (count_include_pad) ++cnt;
    }
  }
  if (cnt == 0) cnt = 1;
  const float g = dY[((n*C + c)*Ho + ho)*Wo + wo] / (float)cnt;

  for (int kh=0; kh<kH; ++kh) {
    const int ih = hstart + kh * dH;
    if (ih < 0 || ih >= H) continue;
    for (int kw=0; kw<kW; ++kw) {
      const int iw = wstart + kw * dW;
      if (iw < 0 || iw >= W) {
        // count_include_pad 가 true 여도 입력 메모리는 없으므로 기여 버림
        continue;
      }
      atomicAdd(&dX[base + ih*W + iw], g);
    }
  }
}

// ====== 런처 ======
namespace ai {

void maxpool2d_fwd_kernel_launcher(const float* X, float* Y, int32_t* Ind,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const int nOut = N*C*Ho*Wo;
  const int BS = 256;
  dim3 grid((nOut + BS - 1)/BS), block(BS);
  maxpool2d_fwd_kernel<<<grid, block, 0, s>>>(
      X, Y, Ind, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode, Ho, Wo);
}

// 되돌린 시그니처 (lib이 기대하는 형태)
void maxpool2d_bwd_kernel_launcher(const float* dY, const int32_t* Ind, float* dX,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, cudaStream_t s)
{
  int Ho, Wo;
  output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode, Ho, Wo);

  const int nOut = N*C*Ho*Wo;
  const int BS = 256;
  dim3 grid((nOut + BS - 1)/BS), block(BS);
  maxpool2d_bwd_kernel<<<grid, block, 0, s>>>(
      dY, Ind, dX,
      N,C,H,W, kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode,
      Ho, Wo);
}


void avgpool2d_fwd_kernel_launcher(const float* X, float* Y,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, bool count_include_pad, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const int nOut = N*C*Ho*Wo;
  const int BS = 256;
  dim3 grid((nOut + BS - 1)/BS), block(BS);
  avgpool2d_fwd_kernel<<<grid, block, 0, s>>>(
      X, Y, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,count_include_pad, Ho, Wo);
}

void avgpool2d_bwd_kernel_launcher(const float* dY, float* dX,
                                   int N,int C,int H,int W,
                                   int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,
                                   bool ceil_mode, bool count_include_pad, cudaStream_t s)
{
  int Ho, Wo; output_dims_2d_host(H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,Ho,Wo);
  const int nOut = N*C*Ho*Wo;
  const int BS = 256;
  dim3 grid((nOut + BS - 1)/BS), block(BS);
  avgpool2d_bwd_kernel<<<grid, block, 0, s>>>(
      dY, dX, N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,ceil_mode,count_include_pad, Ho, Wo);
}

} // namespace ai
