#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "backends/cuda/ops/conv2d/api.hpp"
#include "ai/op_schema.hpp"  // gemm_run 등
#include <cassert>

namespace ai {

__global__ void kadd_kernel(float* A, const float* B, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
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

// ---- kernels: 선언을 정의와 동일한 네임스페이스/시그니처로 맞춤 ----
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

void transpose_kernel_launcher(const float*, const float*, int, int, cudaStream_t);
void transpose_kernel_launcher(const float*, float*,       int, int, cudaStream_t); // 오버로드(LHS 비const)
void reduce_db_kernel_launcher(const float*, float*, int, int, cudaStream_t);




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

  // check outs
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
  float *dCol=nullptr, *dTmp=nullptr, *dColT=nullptr; // dTmp: HWo×K or K×Cout temp
  if (cudaMalloc(&dCol,  sizeof(float)*HWo*K) != cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&dTmp,  sizeof(float)*HWo*std::max(K, Cout)) != cudaSuccess) { cudaFree(dCol); return Status::RuntimeError; }
  if (dW) {
    if (cudaMalloc(&dColT, sizeof(float)*K*HWo) != cudaSuccess) { cudaFree(dCol); cudaFree(dTmp); return Status::RuntimeError; }
  }

  // GEMM attrs
  ai::GemmAttrs g{}; g.act=ai::ActKind::None; g.with_bias=false;

  // zero grads (atomic 안전)
  if (dB) cudaMemsetAsync(dB->data, 0, sizeof(float)*Cout, s);
  if (dW) cudaMemsetAsync(dW->data, 0, sizeof(float)*Cout*Cin*Kh*Kw, s);
  if (dX) cudaMemsetAsync(dX->data, 0, sizeof(float)*N*Cin*H*Wd, s);

  const float* dWptr = static_cast<const float*>(W.data);

  for (int n=0; n<N; ++n) {
    const float* x_n  = static_cast<const float*>(X.data)  + (size_t)n*Cin*H*Wd;
    const float* gy_n = static_cast<const float*>(dY.data) + (size_t)n*Cout*Ho*Wo;

    // im2col(X[n]) → dCol [HWo,K]
    ai::im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd,
      Kh, Kw,
      a.stride_h, a.stride_w,
      a.pad_h, a.pad_w,
      a.dil_h, a.dil_w,
      Ho, Wo,
      s
    );

    // dB: sum over HWo
    if (dB) {
      ai::reduce_db_kernel_launcher(gy_n, static_cast<float*>(dB->data), HWo, Cout, s);
    }

    // dW: (X_col^T @ dY2d) -> [K,Cout] 누적
    if (dW) {
      // transpose dCol -> dColT: [K, HWo]
      ai::transpose_kernel_launcher(dCol, dColT, HWo, K, s);

      // GEMM: [K,HWo] @ [HWo,Cout] -> [K,Cout]
      Tensor tA{dColT,                 {DType::F32,Layout::RowMajor,{K,HWo},   {HWo,1}},  Device::CUDA,0};
      Tensor tB{const_cast<float*>(gy_n), {DType::F32,Layout::RowMajor,{HWo,Cout},{Cout,1}}, Device::CUDA,0};
      Tensor tO{dTmp,                  {DType::F32,Layout::RowMajor,{K,Cout},  {Cout,1}}, Device::CUDA,0};
      int rc = ops::gemm_run(tA, tB, /*bias*/nullptr, tO, g, stream);
      if (rc!=0) { cudaFree(dCol); cudaFree(dTmp); cudaFree(dColT); return Status::RuntimeError; }

      // dW += tO
      
      int total = K * Cout;
      dim3 block(256), grid((total + 255)/256);
      kadd_kernel<<<grid, block, 0, s>>>(static_cast<float*>(dW->data), dTmp, total);
    }

    // dX: (dY2d @ W) -> [HWo,K] → col2im 누적
    if (dX) {
      Tensor tA{const_cast<float*>(gy_n), {DType::F32,Layout::RowMajor,{HWo,Cout},{Cout,1}}, Device::CUDA,0};
      Tensor tB{const_cast<float*>(dWptr), {DType::F32,Layout::RowMajor,{Cout,K}, {K,1}},    Device::CUDA,0};
      Tensor tO{dTmp, {DType::F32,Layout::RowMajor,{HWo,K},{K,1}}, Device::CUDA,0};

      int rc = ops::gemm_run(tA, tB, /*bias*/nullptr, tO, g, stream);
      if (rc!=0) { cudaFree(dCol); cudaFree(dTmp); if (dColT) cudaFree(dColT); return Status::RuntimeError; }

      float* dx_n = static_cast<float*>(dX->data) + (size_t)n*Cin*H*Wd;
      ai::col2im_kernel_launcher(
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

  cudaFree(dCol);
  cudaFree(dTmp);
  if (dColT) cudaFree(dColT);
  return Status::Ok;
}

} // namespace ai
