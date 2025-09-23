#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace {

// ---------- im2col ----------
template<int BS>
__global__ void im2col_kernel(const float* __restrict__ X, float* __restrict__ Col,
                              int Cin, int H, int W,
                              int Kh, int Kw,
                              int sH, int sW,
                              int pH, int pW,
                              int dH, int dW,
                              int Ho, int Wo)
{
  const int HWo = Ho * Wo;
  const int K   = Cin * Kh * Kw;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = HWo * K;
  if (idx >= total) return;

  int k  = idx % K;         // col index
  int hw = idx / K;         // row index

  int w_out = hw % Wo;
  int h_out = hw / Wo;

  int kw = k % Kw; int t = k / Kw;
  int kh = t % Kh; t = t / Kh;
  int c  = t;

  int h_in = h_out * sH - pH + kh * dH;
  int w_in = w_out * sW - pW + kw * dW;

  float v = 0.f;
  if ((unsigned)h_in < (unsigned)H && (unsigned)w_in < (unsigned)W) {
    const float* x_c = X + (size_t)c * H * W;
    v = x_c[h_in * W + w_in];
  }
  Col[hw * K + k] = v;
}

// ---------- col2im ----------
template<int BS>
__global__ void col2im_kernel(const float* __restrict__ Col, float* __restrict__ Xgrad,
                              int Cin, int H, int W,
                              int Kh, int Kw,
                              int sH, int sW,
                              int pH, int pW,
                              int dH, int dW,
                              int Ho, int Wo)
{
  // Xgrad shape: [Cin,H,W]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Cin * H * W;
  if (idx >= total) return;

  int w = idx % W; int t = idx / W;
  int h = t % H;   int c = t / H;

  float acc = 0.f;
  // 역탐색: (h,w)가 어떤 (h_out,w_out,kh,kw)에 의해 쓰였는지
  for (int kh=0; kh<Kh; ++kh){
    int h_out_nom = h + pH - kh * dH;
    if (h_out_nom % sH) continue;
    int h_out = h_out_nom / sH;
    if ((unsigned)h_out >= (unsigned)Ho) continue;

    for (int kw=0; kw<Kw; ++kw){
      int w_out_nom = w + pW - kw * dW;
      if (w_out_nom % sW) continue;
      int w_out = w_out_nom / sW;
      if ((unsigned)w_out >= (unsigned)Wo) continue;

      int HWo = Ho * Wo;
      int K   = Cin * Kh * Kw;
      int k   = (c * Kh + kh) * Kw + kw;
      int hw  = h_out * Wo + w_out;
      acc += Col[hw * K + k];
    }
  }
  Xgrad[idx] += acc; // 누적
}

// ---------- row-major 2D transpose ----------
template<int BS>
__global__ void transpose_kernel(const float* __restrict__ A, float* __restrict__ AT,
                                 int M, int N) // A[M,N] -> AT[N,M]
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) return;
  int n = idx % N;
  int m = idx / N;
  AT[n * M + m] = A[m * N + n];
}

// ---------- dB reduction: sum over [Hout*Wout] with atomics, batch-accum ----------
template<int BS>
__global__ void reduce_db_kernel(const float* __restrict__ dY, float* __restrict__ dB,
                                 int HWo, int Cout)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // over HWo*Cout
  int total = HWo * Cout;
  if (idx >= total) return;
  int co = idx % Cout;
  int hw = idx / Cout;
  atomicAdd(&dB[co], dY[hw * Cout + co]);
}

} // anonymous

namespace ai {

// 런처들
void im2col_kernel_launcher(const float* X, float* Col,
                            int Cin, int H, int W,
                            int Kh, int Kw,
                            int sH, int sW,
                            int pH, int pW,
                            int dH, int dW,
                            int Ho, int Wo,
                            cudaStream_t s)
{
  const int HWo = Ho * Wo;
  const int K   = Cin * Kh * Kw;
  const int total = HWo * K;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  im2col_kernel<BS><<<grid, block, 0, s>>>(X, Col, Cin, H, W, Kh, Kw, sH, sW, pH, pW, dH, dW, Ho, Wo);
}

void col2im_kernel_launcher(const float* Col, float* dX,
                            int Cin, int H, int W,
                            int Kh, int Kw,
                            int sH, int sW,
                            int pH, int pW,
                            int dH, int dW,
                            int Ho, int Wo,
                            cudaStream_t s)
{
  const int total = Cin * H * W;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  col2im_kernel<BS><<<grid, block, 0, s>>>(Col, dX, Cin, H, W, Kh, Kw, sH, sW, pH, pW, dH, dW, Ho, Wo);
}

void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s)
{
  const int total = M * N;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  transpose_kernel<BS><<<grid, block, 0, s>>>(A, AT, M, N);
}

void reduce_db_kernel_launcher(const float* dY, float* dB, int HWo, int Cout, cudaStream_t s)
{
  const int total = HWo * Cout;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  reduce_db_kernel<BS><<<grid, block, 0, s>>>(dY, dB, HWo, Cout);
}

} // namespace ai
