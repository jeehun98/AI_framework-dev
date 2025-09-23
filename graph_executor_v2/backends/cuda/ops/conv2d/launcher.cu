#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "backends/cuda/ops/conv2d/api.hpp"
#include "ai/op_schema.hpp"
#include <cassert>

namespace ai {

// ========================= utils =========================
__global__ void kadd_kernel(float* A, const float* B, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
}

// [Cout, HWo] row-major (채널이 행)로 bias 더하기
__global__ void add_bias_rows(float* __restrict__ Y,      // [Cout, HWo]
                              const float* __restrict__ B, // [Cout]
                              int Cout, int HWo) {
  int hw = blockIdx.x * blockDim.x + threadIdx.x; // 0..HWo-1
  int co = blockIdx.y * blockDim.y + threadIdx.y; // 0..Cout-1
  if (co < Cout && hw < HWo) {
    Y[(size_t)co * HWo + hw] += B[co];
  }
}

static inline bool is4_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is1_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ========================= external launchers (선언) =========================
void im2col_kernel_launcher(const float*, float*,
                            int,int,int,     // Cin,H,W
                            int,int,         // Kh,Kw
                            int,int,         // sH,sW
                            int,int,         // pH,pW
                            int,int,         // dH,dW
                            int,int,         // Ho,Wo
                            cudaStream_t);

void col2im_kernel_launcher(const float*, float*,
                            int,int,int,     // Cin,H,W
                            int,int,         // Kh,Kw
                            int,int,         // sH,sW
                            int,int,         // pH,pW
                            int,int,         // dH,dW
                            int,int,         // Ho,Wo
                            cudaStream_t);

// row-major transpose: in[M,N] -> out[N,M]
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t);

// ===== dB reduce: gy가 [Cout, HWo] 일 때, 행(co) 기준 합산 =====
__global__ void reduce_db_rows_kernel(const float* __restrict__ gy, // [Cout, HWo]
                                      float* __restrict__ db,       // [Cout]
                                      int Cout, int HWo)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Cout * HWo;
  if (idx >= total) return;
  int hw = idx % HWo;
  int co = idx / HWo;
  atomicAdd(&db[co], gy[(size_t)co * HWo + hw]);
}

static inline void reduce_db_rows_kernel_launcher(const float* gy, float* db, int Cout, int HWo, cudaStream_t s){
  const int total = Cout * HWo;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1)/BS);
  reduce_db_rows_kernel<<<grid, block, 0, s>>>(gy, db, Cout, HWo);
}

// ========================= pack/unpack W (K = (ci, kh, kw), kw fastest) =========================
// W: [Cout, Cin, Kh, Kw]
// out_KC: [K, Cout]   row-major (stride {Cout,1})
// out_CK: [Cout, K]   row-major (stride {K,1})

__global__ void pack_w_oihw_to_KC(const float* __restrict__ W,
                                  float* __restrict__ out_KC,
                                  int Cout, int Cin, int Kh, int Kw)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int co  = blockIdx.y;
  int K = Cin * Kh * Kw;
  if (co >= Cout || tid >= K) return;

  // K index: (ci, kh, kw) with kw fastest
  int kw = tid % Kw;
  int tmp= tid / Kw;
  int kh = tmp % Kh;
  int ci = tmp / Kh;

  size_t idx_in  = ((size_t)co * Cin + ci) * Kh * Kw + kh * Kw + kw;   // OIHW
  size_t idx_out = (size_t)tid * Cout + co;                             // [K,Cout]

  out_KC[idx_out] = W[idx_in];
}

__global__ void pack_w_oihw_to_CK(const float* __restrict__ W,
                                  float* __restrict__ out_CK,
                                  int Cout, int Cin, int Kh, int Kw)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int co  = blockIdx.y;
  int K = Cin * Kh * Kw;
  if (co >= Cout || tid >= K) return;

  int kw = tid % Kw;
  int tmp= tid / Kw;
  int kh = tmp % Kh;
  int ci = tmp / Kh;

  size_t idx_in  = ((size_t)co * Cin + ci) * Kh * Kw + kh * Kw + kw;   // OIHW
  size_t idx_out = (size_t)co * K + tid;                                // [Cout,K]

  out_CK[idx_out] = W[idx_in];
}

// dWpack[Cout,K] -> dW[O,I,H,W] add
__global__ void unpack_ck_to_oihw_add(const float* __restrict__ dWpack,
                                      float* __restrict__ dW,
                                      int Cout, int Cin, int Kh, int Kw)
{
  int co = blockIdx.y;
  int tid= blockIdx.x * blockDim.x + threadIdx.x;
  int K = Cin * Kh * Kw;
  if (co >= Cout || tid >= K) return;

  int kw = tid % Kw;
  int tmp= tid / Kw;
  int kh = tmp % Kh;
  int ci = tmp / Kh;

  size_t idx_out = ((size_t)co * Cin + ci) * Kh * Kw + kh * Kw + kw; // OIHW
  size_t idx_in  = (size_t)co * K + tid;                              // [Cout,K]

  dW[idx_out] += dWpack[idx_in];
}

// ======================= Forward =======================
Status Conv2DCudaLaunch(const Tensor& X, const Tensor& W, const Tensor* B, Tensor& Y,
                        const Conv2DAttrs& a, StreamHandle stream)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(Y)) return Status::Invalid;
  if (!is4_f32_cuda(W)) return Status::Invalid;
  if (a.groups != 1) return Status::Unimplemented;

  const int N   = (int)X.desc.shape[0];
  const int Cin = (int)X.desc.shape[1];
  const int H   = (int)X.desc.shape[2];
  const int Wd  = (int)X.desc.shape[3];

  const int Cout= (int)W.desc.shape[0];
  const int WCin= (int)W.desc.shape[1];
  const int Kh  = (int)W.desc.shape[2];
  const int Kw  = (int)W.desc.shape[3];
  if (WCin != Cin) return Status::ShapeMismatch;

  const int Ho = (H  + 2*a.pad_h - a.dil_h*(Kh-1) - 1)/a.stride_h + 1;
  const int Wo = (Wd + 2*a.pad_w - a.dil_w*(Kw-1) - 1)/a.stride_w + 1;

  if (Y.desc.shape[0]!=N || Y.desc.shape[1]!=Cout || Y.desc.shape[2]!=Ho || Y.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;

  const float* dW = static_cast<const float*>(W.data);
  const float* dB = (B && B->data) ? static_cast<const float*>(B->data) : nullptr;

  const int K   = Cin*Kh*Kw;
  const int HWo = Ho*Wo;
  auto s = to_cuda(stream);

  // workspaces
  float *dCol=nullptr, *W_KC=nullptr, *Y_tmp=nullptr;  // Y_tmp: [HWo, Cout]
  if (cudaMalloc(&dCol,  sizeof(float)*HWo*K)      != cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&W_KC,  sizeof(float)*K*Cout)     != cudaSuccess) { cudaFree(dCol); return Status::RuntimeError; }
  if (cudaMalloc(&Y_tmp, sizeof(float)*HWo*Cout)   != cudaSuccess) { cudaFree(dCol); cudaFree(W_KC); return Status::RuntimeError; }

  // pack W → [K, Cout]  (K=(ci,kh,kw), kw fastest)
  {
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_KC<<<grid, block, 0, s>>>(dW, W_KC, Cout, Cin, Kh, Kw);
  }

  // GEMM attrs (bias는 수동 add)
  ai::GemmAttrs g{}; g.act = ai::ActKind::None; g.with_bias = false;

  for (int n=0; n<N; ++n) {
    const float* x_n = static_cast<const float*>(X.data) + (size_t)n*Cin*H*Wd;
    float*       y_n = static_cast<float*>(Y.data)       + (size_t)n*Cout*Ho*Wo;

    // im2col: [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd,
      Kh, Kw,
      a.stride_h, a.stride_w,
      a.pad_h, a.pad_w,
      a.dil_h, a.dil_w,
      Ho, Wo,
      s
    );

    // GEMM: [HWo,K] @ [K,Cout] -> [HWo,Cout] into Y_tmp
    Tensor tA{dCol,  {DType::F32, Layout::RowMajor, {HWo, K},    {K, 1}},     Device::CUDA, 0};
    Tensor tB{W_KC,  {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}},  Device::CUDA, 0};
    Tensor tY{Y_tmp, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}},  Device::CUDA, 0};

    int rc = ops::gemm_run(tA, tB, /*bias*/nullptr, tY, g, stream);
    if (rc!=0) { cudaFree(dCol); cudaFree(W_KC); cudaFree(Y_tmp); return Status::RuntimeError; }

    // transpose: [HWo, Cout] -> [Cout, HWo] directly into y_n (NCHW contiguous)
    transpose_kernel_launcher(Y_tmp, y_n, /*M=*/HWo, /*N=*/Cout, s);

    // bias: row-wise(채널 기준)로 더하기
    if (dB) {
      dim3 block(128, 1);
      dim3 grid((HWo + block.x - 1)/block.x, Cout);
      add_bias_rows<<<grid, block, 0, s>>>(y_n, dB, Cout, HWo);
    }
  }

  cudaFree(dCol);
  cudaFree(W_KC);
  cudaFree(Y_tmp);
  return Status::Ok;
}

// ======================= Backward =======================
Status Conv2DCudaBackwardLaunch(const Tensor& X, const Tensor& W, const Tensor& dY,
                                Tensor* dW, Tensor* dB, Tensor* dX,
                                const Conv2DAttrs& a, StreamHandle stream)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(W) || !is4_f32_cuda(dY)) return Status::Invalid;
  if (a.groups != 1) return Status::Unimplemented;

  const int N   = (int)X.desc.shape[0];
  const int Cin = (int)X.desc.shape[1];
  const int H   = (int)X.desc.shape[2];
  const int Wd  = (int)X.desc.shape[3];

  const int Cout= (int)W.desc.shape[0];
  const int WCin= (int)W.desc.shape[1];
  const int Kh  = (int)W.desc.shape[2];
  const int Kw  = (int)W.desc.shape[3];
  if (WCin != Cin) return Status::ShapeMismatch;

  const int Ho = (H  + 2*a.pad_h - a.dil_h*(Kh-1) - 1)/a.stride_h + 1;
  const int Wo = (Wd + 2*a.pad_w - a.dil_w*(Kw-1) - 1)/a.stride_w + 1;

  if (dY.desc.shape[0]!=N || dY.desc.shape[1]!=Cout || dY.desc.shape[2]!=Ho || dY.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;

  // outs check
  if (dW) {
    if (!is4_f32_cuda(*dW) ||
        dW->desc.shape[0]!=Cout || dW->desc.shape[1]!=Cin ||
        dW->desc.shape[2]!=Kh   || dW->desc.shape[3]!=Kw) return Status::ShapeMismatch;
  }
  if (dB) {
    if (!(is1_f32_cuda(*dB) && (int)dB->desc.shape[0]==Cout)) return Status::ShapeMismatch;
  }
  if (dX) {
    if (!is4_f32_cuda(*dX) ||
        dX->desc.shape[0]!=N || dX->desc.shape[1]!=Cin ||
        dX->desc.shape[2]!=H || dX->desc.shape[3]!=Wd) return Status::ShapeMismatch;
  }

  const int K   = Cin*Kh*Kw;
  const int HWo = Ho*Wo;
  auto s = to_cuda(stream);

  // workspaces
  float *dCol=nullptr, *dTmp=nullptr, *W_CK=nullptr, *dWpack=nullptr, *dY_HT=nullptr;
  if (cudaMalloc(&dCol, sizeof(float)*HWo*K) != cudaSuccess) return Status::RuntimeError;

  size_t tmp_elems = (size_t)std::max(Cout*K, HWo*K);
  if (cudaMalloc(&dTmp, sizeof(float)*tmp_elems) != cudaSuccess) { cudaFree(dCol); return Status::RuntimeError; }

  // grads zero
  if (dB) cudaMemsetAsync(dB->data, 0, sizeof(float)*Cout, s);
  if (dW) cudaMemsetAsync(dW->data, 0, sizeof(float)*Cout*Cin*Kh*Kw, s);
  if (dX) cudaMemsetAsync(dX->data, 0, sizeof(float)*N*Cin*H*Wd, s);

  // dX용 W_CK: [Cout, K]  (K=(ci,kh,kw), kw fastest)
  if (dX) {
    const float* dWsrc = static_cast<const float*>(W.data);
    if (cudaMalloc(&W_CK, sizeof(float)*Cout*K) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp);
      return Status::RuntimeError;
    }
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_CK<<<grid, block, 0, s>>>(dWsrc, W_CK, Cout, Cin, Kh, Kw);
    // dY_HT: [HWo, Cout] (gy_n의 전치 보관용)
    if (cudaMalloc(&dY_HT, sizeof(float)*HWo*Cout) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp); cudaFree(W_CK);
      return Status::RuntimeError;
    }
  }

  // dW 누적 버퍼: [Cout, K]
  if (dW) {
    if (cudaMalloc(&dWpack, sizeof(float)*Cout*K) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT);
      return Status::RuntimeError;
    }
    cudaMemsetAsync(dWpack, 0, sizeof(float)*Cout*K, s);
  }

  // GEMM attrs
  ai::GemmAttrs g{}; g.act=ai::ActKind::None; g.with_bias=false;

  for (int n=0; n<N; ++n) {
    const float* x_n  = static_cast<const float*>(X.data)  + (size_t)n*Cin*H*Wd;
    const float* gy_n = static_cast<const float*>(dY.data) + (size_t)n*Cout*Ho*Wo; // [Cout, HWo]

    // im2col(X[n]) → dCol [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd,
      Kh, Kw,
      a.stride_h, a.stride_w,
      a.pad_h, a.pad_w,
      a.dil_h, a.dil_w,
      Ho, Wo,
      s
    );

    // dB: sum over HWo (gy_n is [Cout, HWo])
    if (dB) {
      reduce_db_rows_kernel_launcher(gy_n, static_cast<float*>(dB->data), Cout, HWo, s);
    }

    // dW: dWpack += gy_n[Cout,HWo] @ X_col[HWo, K] -> [Cout, K]
    if (dW) {
      Tensor tA{const_cast<float*>(gy_n), {DType::F32, Layout::RowMajor, {Cout, HWo}, {HWo, 1}}, Device::CUDA, 0};
      Tensor tB{dCol,                     {DType::F32, Layout::RowMajor, {HWo,  K},   {K,   1}}, Device::CUDA, 0};
      Tensor tO{dTmp,                     {DType::F32, Layout::RowMajor, {Cout, K},   {K,   1}}, Device::CUDA, 0};

      int rc = ops::gemm_run(tA, tB, /*bias*/nullptr, tO, g, stream);
      if (rc!=0) { cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dWpack) cudaFree(dWpack); if (dY_HT) cudaFree(dY_HT); return Status::RuntimeError; }

      int total = Cout * K;
      dim3 block(256), grid((total + 255)/256);
      kadd_kernel<<<grid, block, 0, s>>>(dWpack, dTmp, total);
    }

    // dX: dX += (dY[HWo,Cout] @ W_CK[Cout,K]) -> [HWo,K] -> col2im
    if (dX) {
      // gy_n: [Cout, HWo] → transpose to [HWo, Cout]
      transpose_kernel_launcher(gy_n, dY_HT, /*M=*/Cout, /*N=*/HWo, s);

      Tensor tA{dY_HT, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};
      Tensor tB{W_CK,  {DType::F32, Layout::RowMajor, {Cout, K},   {K,    1}}, Device::CUDA, 0};
      Tensor tO{dTmp,  {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0};

      int rc = ops::gemm_run(tA, tB, /*bias*/nullptr, tO, g, stream);
      if (rc!=0) { cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dWpack) cudaFree(dWpack); if (dY_HT) cudaFree(dY_HT); return Status::RuntimeError; }

      float* dx_n = static_cast<float*>(dX->data) + (size_t)n*Cin*H*Wd;
      col2im_kernel_launcher(
        dTmp, dx_n,
        Cin, H, Wd,
        Kh, Kw,
        a.stride_h, a.stride_w,
        a.pad_h, a.pad_w,
        a.dil_h, a.dil_w,
        Ho, Wo,
        s
      );
    }
  }

  // dW 최종 반영: dWpack[Cout,K] -> dW[O,I,H,W]
  if (dW) {
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    unpack_ck_to_oihw_add<<<grid, block, 0, s>>>(dWpack,
      static_cast<float*>(dW->data), Cout, Cin, Kh, Kw);
  }

  cudaFree(dCol);
  cudaFree(dTmp);
  if (W_CK)   cudaFree(W_CK);
  if (dWpack) cudaFree(dWpack);
  if (dY_HT)  cudaFree(dY_HT);
  return Status::Ok;
}

} // namespace ai
