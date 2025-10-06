// backends/cuda/ops/conv2d/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace { // ---- device 템플릿 커널들: TU 내부 전용 ----

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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Cin * H * W;
  if (idx >= total) return;

  int w = idx % W; int t = idx / W;
  int h = t % H;   int c = t / H;

  float acc = 0.f;
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
  Xgrad[idx] += acc;
}

// ---------- row-major 2D transpose ----------
template<int BS>
__global__ void transpose_kernel(const float* __restrict__ A, float* __restrict__ AT,
                                 int M, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) return;
  int n = idx % N;
  int m = idx / N;
  AT[n * M + m] = A[m * N + n];
}

// ---------- dB reduction ----------
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

} // anonymous TU-local

namespace ai { // ---- 외부에서 보이는 심볼들 ----

// ===== pack/unpack 정의 (외부 링크 필요: __global__ 심볼을 ai 네임스페이스로) =====
__global__ void pack_w_oihw_to_KC(const float* __restrict__ W,
                                  float* __restrict__ out_KC,
                                  int Cout, int Cin, int Kh, int Kw)
{
  const int K = Cin * Kh * Kw;
  int k  = blockIdx.x * blockDim.x + threadIdx.x; // 0..K-1
  int co = blockIdx.y;                             // 0..Cout-1
  if (k >= K || co >= Cout) return;

  int tmp = k;
  int kw = tmp % Kw; tmp /= Kw;
  int kh = tmp % Kh; tmp /= Kh;
  int c  = tmp;

  size_t off = (((size_t)co * Cin + c) * Kh + kh) * Kw + kw; // OIHW
  out_KC[(size_t)k * Cout + co] = W[off];                    // [K, Cout]
}

__global__ void pack_w_oihw_to_CK(const float* __restrict__ W,
                                  float* __restrict__ out_CK,
                                  int Cout, int Cin, int Kh, int Kw)
{
  const int K = Cin * Kh * Kw;
  int k  = blockIdx.x * blockDim.x + threadIdx.x;
  int co = blockIdx.y;
  if (k >= K || co >= Cout) return;

  int tmp = k;
  int kw = tmp % Kw; tmp /= Kw;
  int kh = tmp % Kh; tmp /= Kh;
  int c  = tmp;

  size_t off = (((size_t)co * Cin + c) * Kh + kh) * Kw + kw; // OIHW
  out_CK[(size_t)co * K + k] = W[off];                       // [Cout, K]
}

__global__ void unpack_ck_to_oihw_add(const float* __restrict__ dWpack,
                                      float* __restrict__ dW,
                                      int Cout, int Cin, int Kh, int Kw)
{
  const int K = Cin * Kh * Kw;
  int k  = blockIdx.x * blockDim.x + threadIdx.x;
  int co = blockIdx.y;
  if (k >= K || co >= Cout) return;

  int tmp = k;
  int kw = tmp % Kw; tmp /= Kw;
  int kh = tmp % Kh; tmp /= Kh;
  int c  = tmp;

  size_t off = (((size_t)co * Cin + c) * Kh + kh) * Kw + kw; // OIHW
  dW[off] += dWpack[(size_t)co * K + k];                     // 누적
}

// ===== 런처들 =====
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

// 호출처(launcher.cu)와 이름/인자 순서 맞춤!
void reduce_db_rows_kernel_launcher(const float* gy, float* db, int Cout, int HWo, cudaStream_t s)
{
  const int total = HWo * Cout;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  // reduce_db_kernel은 (dY, dB, HWo, Cout) 순으로 받음
  reduce_db_kernel<BS><<<grid, block, 0, s>>>(gy, db, HWo, Cout);
}

} // namespace ai
