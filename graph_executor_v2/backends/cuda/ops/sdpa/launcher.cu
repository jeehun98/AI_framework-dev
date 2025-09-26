// backends/cuda/ops/sdpa/launcher.cu
#include <cuda_runtime.h>
#include <cmath>
#include "backends/cuda/ops/sdpa/api.hpp"
#include "backends/cuda/ops/dropout/api.hpp"
#include "backends/cuda/ops/softmax/api.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp"
#include <cstdint>

namespace ai { namespace ops {
  // GEMM
  int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
               Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);

  // Softmax (attrs 기반)
  int softmax_run(const Tensor& X, const Tensor* mask, Tensor& Y,
                  const ai::SoftmaxAttrs& attrs, StreamHandle stream);
  int softmax_backward_run(const Tensor& Y, const Tensor& dY, Tensor& dX,
                           const ai::SoftmaxAttrs& attrs, StreamHandle stream);

  // Dropout
  int dropout_run(const Tensor& X, Tensor& Y, Tensor* mask,
                  const ai::DropoutAttrs& attrs, StreamHandle stream);
}}

namespace ai {

static inline bool is_bhxd_f32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// RowMajor 2D transpose: in[R,C] -> out[C,R]
__global__ void transpose_rm_f32(const float* __restrict__ in, float* __restrict__ out,
                                 int R, int C){
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < R && c < C) out[c * R + r] = in[r * C + c];
}

// ---- 로컬 유틸: 4D F32 CUDA 텐서 확인 (RowMajor) ----
static inline bool is4d_f32_cuda(const ai::Tensor& t){
  return t.device==ai::Device::CUDA &&
         t.desc.dtype==ai::DType::F32 &&
         t.desc.layout==ai::Layout::RowMajor &&
         t.desc.shape.size()==4;
}

// causal mask: S[b,h,m,n] += huge_neg if n>m
__global__ void causal_mask_add_kernel(float* S, int B,int H,int M,int N, float huge_neg){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*M*N;
  if (idx >= total) return;
  int t = idx;
  int n = t % N; t /= N;
  int m = t % M; t /= M;
  if (n > m) S[idx] += huge_neg;
}

// zero gS on upper triangle
__global__ void causal_gs_zero_kernel(float* gS, int B,int H,int M,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*M*N;
  if (idx >= total) return;
  int t = idx;
  int n = t % N; t /= N;
  int m = t % M; t /= M;
  if (n > m) gS[idx] = 0.f;
}

__global__ void scale_kernel(float* X, int64_t n, float s){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) X[i] *= s;
}

// mask: I8/I32/F32 지원. 
__global__ void add_mask_i8_kernel(float* S, const int8_t* M, int B,int H,int Mlen,int N, float huge_neg){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  if (M[idx]) S[idx] += huge_neg;
}
__global__ void add_mask_i32_kernel(float* S, const int32_t* M, int B,int H,int Mlen,int N, float huge_neg){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  if (M[idx]) S[idx] += huge_neg;
}
__global__ void add_mask_f32_kernel(float* S, const float* M, int B,int H,int Mlen,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  S[idx] += M[idx];
}

// backward에서 gS를 마스크 위치에 0으로
__global__ void zero_gs_mask_i8_kernel(float* gS, const int8_t* M, int B,int H,int Mlen,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  if (M[idx]) gS[idx] = 0.f;
}
__global__ void zero_gs_mask_i32_kernel(float* gS, const int32_t* M, int B,int H,int Mlen,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  if (M[idx]) gS[idx] = 0.f;
}
__global__ void zero_gs_mask_f32_kernel(float* gS, const float* M, int B,int H,int Mlen,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B*H*Mlen*N;
  if (idx >= total) return;
  if (M[idx] != 0.0f) gS[idx] = 0.f;
}

// =============== Forward ===============
Status SDPACudaLaunch(const Tensor& Q, const Tensor& K, const Tensor& V,
                      const Tensor* mask, Tensor& Y,
                      const SDPAAttrs& attrs, StreamHandle stream)
{
  if (!is_bhxd_f32_4d_cuda(Q) || !is_bhxd_f32_4d_cuda(K) ||
      !is_bhxd_f32_4d_cuda(V) || !is_bhxd_f32_4d_cuda(Y))
    return Status::Invalid;

  const int B  = (int)Q.desc.shape[0];
  const int H  = (int)Q.desc.shape[1];
  const int M  = (int)Q.desc.shape[2];
  const int D  = (int)Q.desc.shape[3];
  const int N  = (int)K.desc.shape[2];

  if (K.desc.shape[0]!=B || K.desc.shape[1]!=H || K.desc.shape[3]!=D) return Status::ShapeMismatch;
  if (V.desc.shape[0]!=B || V.desc.shape[1]!=H || V.desc.shape[2]!=N || V.desc.shape[3]!=D) return Status::ShapeMismatch;
  if (Y.desc.shape[0]!=B || Y.desc.shape[1]!=H || Y.desc.shape[2]!=M  || Y.desc.shape[3]!=D) return Status::ShapeMismatch;

  // workspace
  size_t nScores = (size_t)B*H*M*N;
  float *dS=nullptr, *dP=nullptr, *dKt=nullptr;
  if (cudaMalloc(&dS,  sizeof(float)*nScores)!=cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&dP,  sizeof(float)*nScores)!=cudaSuccess){ cudaFree(dS); return Status::RuntimeError; }
  if (cudaMalloc(&dKt, sizeof(float)*(size_t)D*(size_t)N)!=cudaSuccess){ cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

  GemmAttrs g{}; g.act=ActKind::None; g.with_bias=false;
  const float scale = (attrs.scale!=0.f) ? attrs.scale : (1.f/std::sqrt((float)D));
  cudaStream_t s = to_cuda(stream);

  auto slice2d = [](const Tensor& T, int b, int h, int R, int C)->Tensor{
    size_t offset = ((size_t)b*T.desc.shape[1] + h) * (size_t)R*C;
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={R,C}; d.stride={C,1};
    return Tensor{ (void*)((float*)T.data + offset), d, Device::CUDA, T.device_index };
  };

  Tensor S4{ dS, {DType::F32, Layout::RowMajor, {B,H,M,N}, {H*M*N, M*N, N, 1}}, Device::CUDA, Q.device_index };
  Tensor P4{ dP, {DType::F32, Layout::RowMajor, {B,H,M,N}, {H*M*N, M*N, N, 1}}, Device::CUDA, Q.device_index };

  // ---- Step 1: S = Q @ K^T ----
  dim3 blk(32, 8), grdKT((N + blk.x - 1)/blk.x, (D + blk.y - 1)/blk.y);

  for (int b=0;b<B;++b){
    for (int h=0; h<H; ++h){
      Tensor Q2 = slice2d(Q, b,h,M,D);
      Tensor K2 = slice2d(K, b,h,N,D);

      // K(N,D) -> Kt(D,N)
      const float* Kptr = static_cast<const float*>(K2.data);
      transpose_rm_f32<<<grdKT, blk, 0, s>>>(Kptr, dKt, /*R*/N, /*C*/D);
      if (cudaPeekAtLastError()!=cudaSuccess){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

      TensorDesc kt_d{}; kt_d.dtype=DType::F32; kt_d.layout=Layout::RowMajor; kt_d.shape={D,N}; kt_d.stride={N,1};
      Tensor Kt{ dKt, kt_d, Device::CUDA, Q.device_index };

      // S = Q @ Kt
      Tensor S2 = slice2d(S4, b,h,M,N);
      if (ops::gemm_run(Q2, Kt, nullptr, S2, g, stream)!=0){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

      // scale: S *= scale
      long long n = (long long)M*(long long)N; int BS=256, GRID=(int)((n+BS-1)/BS);
      scale_kernel<<<GRID,BS,0,s>>>((float*)S2.data, n, scale);
      if (cudaPeekAtLastError()!=cudaSuccess){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }
    }
  }

  // ---- Step 2: causal mask ----
  if (attrs.causal){
    int BS = 256;
    size_t total = (size_t)B * H * M * N;
    int GRID = (int)((total + BS - 1) / BS);
    causal_mask_add_kernel<<<GRID, BS, 0, s>>>(dS, B, H, M, N, -1e9f);
    if (cudaPeekAtLastError()!=cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP);
      return Status::RuntimeError;
    }
  }

  // ---- Step 2.5: external mask ----
  if (mask) {
    if (mask->desc.shape.size()!=4 ||
        (int)mask->desc.shape[0]!=B || (int)mask->desc.shape[1]!=1 ||
        (int)mask->desc.shape[2]!=M || (int)mask->desc.shape[3]!=N) {
      cudaFree(dKt); cudaFree(dS); cudaFree(dP);
      return Status::ShapeMismatch;
    }

    int BS = 256;
    size_t total = (size_t)B * H * M * N;
    int GRID = (int)((total + BS - 1) / BS);

    switch (mask->desc.dtype) {
      case DType::I8:
        add_mask_i8_kernel<<<GRID, BS, 0, s>>>(dS, (const int8_t*)mask->data, B, H, M, N, -1e9f);
        break;
      case DType::I32:
        add_mask_i32_kernel<<<GRID, BS, 0, s>>>(dS, (const int32_t*)mask->data, B, H, M, N, -1e9f);
        break;
      case DType::F32:
        add_mask_f32_kernel<<<GRID, BS, 0, s>>>(dS, (const float*)mask->data, B, H, M, N);
        break;
      default:
        cudaFree(dKt); cudaFree(dS); cudaFree(dP);
        return Status::Invalid;
    }
    if (cudaPeekAtLastError()!=cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP);
      return Status::RuntimeError;
    }
  }

  // ---- Step 3/4: P = softmax(S) [+ dropout] ----
  {
    SoftmaxAttrs sa{}; sa.scale=1.f; sa.log=false;
    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        Tensor S2{ (char*)dS + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                   {DType::F32, Layout::RowMajor, {M,N}, {N,1}}, Device::CUDA, Q.device_index };
        Tensor P2{ (char*)dP + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                   {DType::F32, Layout::RowMajor, {M,N}, {N,1}}, Device::CUDA, Q.device_index };
        if (ops::softmax_run(S2, /*mask*/nullptr, P2, sa, stream)!=0){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

        if (attrs.dropout_p > 0.f){
          DropoutAttrs da{}; da.p=attrs.dropout_p; da.scale_in_train=attrs.scale_in_train; da.seed=attrs.seed;
          if (ops::dropout_run(P2, P2, /*mask*/nullptr, da, stream)!=0){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }
        }

        // ---- Step 5: Y = P @ V ----
        Tensor V2 = slice2d(V, b,h,N,D);
        Tensor Y2 = slice2d(Y, b,h,M,D);
        if (ops::gemm_run(P2, V2, nullptr, Y2, g, stream)!=0){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }
      }
    }
  }

  cudaFree(dKt); cudaFree(dS); cudaFree(dP);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// 런타임 에러 리턴 헬퍼: 태그와 CUDA 에러 문자열(있다면) 출력
static inline ai::Status RTERR(const char* tag, cudaError_t err = cudaSuccess) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[SDPA-BWD][RuntimeError] at %s | cuda: %s\n", tag, cudaGetErrorString(err));
  } else {
    fprintf(stderr, "[SDPA-BWD][RuntimeError] at %s\n", tag);
  }
  return ai::Status::RuntimeError;
}

ai::Status SDPACudaBackwardLaunch(const ai::Tensor& Q, const ai::Tensor& K, const ai::Tensor& V,
                                  const ai::Tensor& dY,
                                  const ai::Tensor* mask, 
                                  ai::Tensor* dQ, ai::Tensor* dK, ai::Tensor* dV,
                                  const ai::SDPAAttrs& a, ai::StreamHandle stream)
{
  // 최소 정책: 셋 다 null이면 안 됨
  if (!dQ && !dK && !dV) { fprintf(stderr, "[SDPA-BWD] invalid: all outputs null\n"); return ai::Status::Invalid; }

  // 타입/차원 체크
  if (!is4d_f32_cuda(Q) || !is4d_f32_cuda(K) || !is4d_f32_cuda(V) || !is4d_f32_cuda(dY)) {
    fprintf(stderr, "[SDPA-BWD] invalid: dtype/layout/device/ndim check failed for inputs\n");
    return ai::Status::Invalid;
  }
  if (dQ && !is4d_f32_cuda(*dQ)) { fprintf(stderr, "[SDPA-BWD] invalid: dQ bad\n"); return ai::Status::Invalid; }
  if (dK && !is4d_f32_cuda(*dK)) { fprintf(stderr, "[SDPA-BWD] invalid: dK bad\n"); return ai::Status::Invalid; }
  if (dV && !is4d_f32_cuda(*dV)) { fprintf(stderr, "[SDPA-BWD] invalid: dV bad\n"); return ai::Status::Invalid; }

  const int B = (int)Q.desc.shape[0];
  const int H = (int)Q.desc.shape[1];
  const int M = (int)Q.desc.shape[2];
  const int D = (int)Q.desc.shape[3];
  const int N = (int)K.desc.shape[2];

  if ((int)K.desc.shape[0]!=B || (int)K.desc.shape[1]!=H || (int)K.desc.shape[3]!=D) {
    fprintf(stderr, "[SDPA-BWD] shape mismatch: K\n"); return ai::Status::ShapeMismatch;
  }
  if ((int)V.desc.shape[0]!=B || (int)V.desc.shape[1]!=H || (int)V.desc.shape[2]!=N || (int)V.desc.shape[3]!=D) {
    fprintf(stderr, "[SDPA-BWD] shape mismatch: V\n"); return ai::Status::ShapeMismatch;
  }
  if ((int)dY.desc.shape[0]!=B || (int)dY.desc.shape[1]!=H || (int)dY.desc.shape[2]!=M || (int)dY.desc.shape[3]!=D) {
    fprintf(stderr, "[SDPA-BWD] shape mismatch: dY\n"); return ai::Status::ShapeMismatch;
  }
  if (dQ && dQ->desc.shape!=Q.desc.shape) { fprintf(stderr, "[SDPA-BWD] shape mismatch: dQ vs Q\n"); return ai::Status::ShapeMismatch; }
  if (dK && dK->desc.shape!=K.desc.shape) { fprintf(stderr, "[SDPA-BWD] shape mismatch: dK vs K\n"); return ai::Status::ShapeMismatch; }
  if (dV && dV->desc.shape!=V.desc.shape) { fprintf(stderr, "[SDPA-BWD] shape mismatch: dV vs V\n"); return ai::Status::ShapeMismatch; }

  if (a.dropout_p != 0.f) { fprintf(stderr, "[SDPA-BWD] invalid: dropout not supported yet\n"); return ai::Status::Invalid; }

  cudaStream_t s = to_cuda(stream);

  // workspace: S,P,gP,gS (all [B,H,M,N])
  size_t nScores = (size_t)B*H*M*N;
  float *dS=nullptr, *dP=nullptr, *dgP=nullptr, *dgS=nullptr;
  if (cudaMalloc(&dS,  sizeof(float)*nScores)!=cudaSuccess) return RTERR("cudaMalloc dS");
  if (cudaMalloc(&dP,  sizeof(float)*nScores)!=cudaSuccess){ cudaFree(dS); return RTERR("cudaMalloc dP"); }
  if (cudaMalloc(&dgP, sizeof(float)*nScores)!=cudaSuccess){ cudaFree(dP); cudaFree(dS); return RTERR("cudaMalloc dgP"); }
  if (cudaMalloc(&dgS, sizeof(float)*nScores)!=cudaSuccess){ cudaFree(dgP); cudaFree(dP); cudaFree(dS); return RTERR("cudaMalloc dgS"); }

  ai::GemmAttrs g{}; g.act=ai::ActKind::None; g.with_bias=false;
  float scale = a.scale; if (scale==0.f) scale = 1.f / std::sqrt((float)D);

  // 공용 전치 버퍼 (Kᵀ / Vᵀ 용)
  float* dKt = nullptr;
  if (cudaMalloc(&dKt, sizeof(float) * (size_t)D * (size_t)N) != cudaSuccess) {
    cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
    return RTERR("cudaMalloc K^T (step1)");
  }
  
  // 1) S = scale * (Q @ K^T)
  for (int b=0; b<B; ++b){
    for (int h=0; h<H; ++h){
      ai::Tensor tQ{ (char*)Q.data + ((size_t)((((size_t)b*H+h)*M)*D))*sizeof(float),
                     {ai::DType::F32, ai::Layout::RowMajor, {M,D}, {D,1}}, ai::Device::CUDA, Q.device_index };
      ai::Tensor tK{ (char*)K.data + ((size_t)((((size_t)b*H+h)*N)*D))*sizeof(float),
                     {ai::DType::F32, ai::Layout::RowMajor, {N,D}, {D,1}}, ai::Device::CUDA, K.device_index };
      ai::Tensor tS{ (char*)dS  + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                     {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };

      // (N×D) -> (D×N)
      {
        dim3 blk(32, 8);
        dim3 grd((D + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);
        transpose_rm_f32<<<grd, blk, 0, s>>>((const float*)tK.data, dKt, /*R*/N, /*C*/D);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("transpose K->Kt (step1)", cudaGetLastError());
        }
      }
      ai::Tensor tKt{ dKt,
                      {ai::DType::F32, ai::Layout::RowMajor, {D,N}, {N,1}},
                      ai::Device::CUDA, K.device_index };

      if (ai::ops::gemm_run(tQ, tKt, nullptr, tS, g, stream)!=0){
        cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
        return RTERR("gemm S = Q @ K^T");
      }

      long long n = (long long)M*(long long)N; int BS=256, GRID=(int)((n+BS-1)/BS);
      scale_kernel<<<GRID,BS,0,s>>>((float*)tS.data, n, scale);
      if (cudaPeekAtLastError()!=cudaSuccess){
        cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
        return RTERR("scale S (step1)", cudaGetLastError());
      }
    }
  }

  // 2) causal mask
  if (a.causal){
    int BS=256, GRID=(int)((nScores + BS - 1)/BS);
    causal_mask_add_kernel<<<GRID,BS,0,s>>>(dS, B,H,M,N, -1e9f);
    if (cudaPeekAtLastError()!=cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("causal_mask_add_kernel", cudaGetLastError());
    }
  }

  // 3) P = softmax(S)
  {
    ai::SoftmaxAttrs sa{}; sa.scale=1.f; sa.log=false;
    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        ai::Tensor tS{ (char*)dS + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                       {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tP{ (char*)dP + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                       {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        if (ai::ops::softmax_run(tS, /*mask*/nullptr, tP, sa, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("softmax forward P");
        }
      }
    }
  }

  // 4) dV = P^T @ dY
  if (dV){
    float* dYt_buf = nullptr;
    if (cudaMalloc(&dYt_buf, sizeof(float) * (size_t)D * (size_t)M) != cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("cudaMalloc dYt_buf (dV path)");
    }

    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        ai::Tensor tP{  (char*)dP + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tdY{ (char*)dY.data + ((size_t)((((size_t)b*H+h)*M)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,D}, {D,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tdV{ (char*)dV->data + ((size_t)((((size_t)b*H+h)*N)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {N,D}, {D,1}}, ai::Device::CUDA, dV->device_index };

        // (M×D) -> (D×M)
        {
          dim3 blk(32,8);
          dim3 grd((D + blk.x - 1)/blk.x, (M + blk.y - 1)/blk.y);
          transpose_rm_f32<<<grd, blk, 0, s>>>((const float*)tdY.data, dYt_buf, /*R*/M, /*C*/D);
          if (cudaPeekAtLastError()!=cudaSuccess){
            cudaFree(dYt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
            return RTERR("transpose dY->dYt (dV path)", cudaGetLastError());
          }
        }
        ai::Tensor tYt{ dYt_buf, {ai::DType::F32, ai::Layout::RowMajor, {D,M}, {M,1}}, ai::Device::CUDA, Q.device_index };

        // T1 = dY^T(D×M) @ P(M×N) = (D×N)
        float* dT1=nullptr;
        if (cudaMalloc(&dT1, sizeof(float)*(size_t)D*(size_t)N)!=cudaSuccess){
          cudaFree(dYt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("cudaMalloc T1 (dV path)");
        }
        ai::Tensor tT1{ dT1, {ai::DType::F32, ai::Layout::RowMajor, {D,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        if (ai::ops::gemm_run(tYt, tP, nullptr, tT1, g, stream)!=0){
          cudaFree(dT1); cudaFree(dYt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("gemm T1 = dY^T @ P");
        }

        // tdV = T1^T
        dim3 blk2(32,8), grd2((N + blk2.x -1)/blk2.x, (D + blk2.y -1)/blk2.y);
        transpose_rm_f32<<<grd2,blk2,0,s>>>(dT1, (float*)tdV.data, /*R*/D, /*C*/N);
        cudaFree(dT1);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(dYt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("transpose tdV", cudaGetLastError());
        }
      }
    }
    cudaFree(dYt_buf);
  }

  // 5) gP = dY @ V^T
  for (int b=0; b<B; ++b){
    for (int h=0; h<H; ++h){
      ai::Tensor tdY{ (char*)dY.data + ((size_t)((((size_t)b*H+h)*M)*D))*sizeof(float),
                      {ai::DType::F32, ai::Layout::RowMajor, {M,D}, {D,1}}, ai::Device::CUDA, Q.device_index };
      ai::Tensor tV { (char*)V.data  + ((size_t)((((size_t)b*H+h)*N)*D))*sizeof(float),
                      {ai::DType::F32, ai::Layout::RowMajor, {N,D}, {D,1}}, ai::Device::CUDA, V.device_index };
      ai::Tensor tgP{ (char*)dgP     + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                      {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };

      // V(N×D) -> Vt(D×N) in dKt (재사용)
      {
        dim3 blk(32, 8);
        dim3 grd((D + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);
        transpose_rm_f32<<<grd, blk, 0, s>>>((const float*)tV.data, dKt, /*R*/N, /*C*/D);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("transpose V->Vt (step5)", cudaGetLastError());
        }
      }
      ai::Tensor tVt{ dKt,
                      {ai::DType::F32, ai::Layout::RowMajor, {D,N}, {N,1}},
                      ai::Device::CUDA, V.device_index };

      if (ai::ops::gemm_run(tdY, tVt, nullptr, tgP, g, stream)!=0){
        cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
        return RTERR("gemm gP = dY @ V^T");
      }
    }
  }

  // 6) gS = softmax_backward(P, gP)
  {
    ai::SoftmaxAttrs sa{}; sa.scale=1.f; sa.log=false;
    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        ai::Tensor tP { (char*)dP  + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tgP{ (char*)dgP + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tgS{ (char*)dgS + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        if (ai::ops::softmax_backward_run(tP, tgP, tgS, sa, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("softmax backward gS");
        }
      }
    }
  }

  // 7) causal ⇒ gS 상삼각 0
  if (a.causal){
    int BS=256, GRID=(int)((nScores + BS - 1)/BS);
    causal_gs_zero_kernel<<<GRID,BS,0,s>>>(dgS, B,H,M,N);
    if (cudaPeekAtLastError()!=cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("causal_gs_zero_kernel", cudaGetLastError());
    }
  }

    // 7.5) external mask ⇒ gS = 0 on masked positions
  if (mask) {
    if (mask->desc.shape.size()!=4 ||
        (int)mask->desc.shape[0]!=B || (int)mask->desc.shape[1]!=1 ||
        (int)mask->desc.shape[2]!=M || (int)mask->desc.shape[3]!=N) {
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("mask shape mismatch (bwd)");
    }
    int BS=256, GRID=(int)(((size_t)B*H*M*N + BS - 1)/BS);
    switch (mask->desc.dtype) {
      case DType::I8:
        zero_gs_mask_i8_kernel<<<GRID,BS,0,s>>>(dgS, (const int8_t*)mask->data, B,H,M,N);
        break;
      case DType::I32:
        zero_gs_mask_i32_kernel<<<GRID,BS,0,s>>>(dgS, (const int32_t*)mask->data, B,H,M,N);
        break;
      case DType::F32:
        zero_gs_mask_f32_kernel<<<GRID,BS,0,s>>>(dgS, (const float*)mask->data, B,H,M,N);
        break;
      default:
        cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
        return RTERR("mask dtype invalid (bwd)");
    }
    if (cudaPeekAtLastError()!=cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("zero_gs_mask_*_kernel", cudaGetLastError());
    }
  }

  // 8) dQ = scale * (gS @ K)
  if (dQ){
    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        ai::Tensor tgS{ (char*)dgS + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tK { (char*)K.data + ((size_t)((((size_t)b*H+h)*N)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {N,D}, {D,1}}, ai::Device::CUDA, K.device_index };
        ai::Tensor tdQ{ (char*)dQ->data + ((size_t)((((size_t)b*H+h)*M)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,D}, {D,1}}, ai::Device::CUDA, dQ->device_index };
        if (ai::ops::gemm_run(tgS, tK, nullptr, tdQ, g, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("gemm dQ = gS @ K");
        }
        long long n = (long long)M*(long long)D; int BS=256, GRID=(int)((n+BS-1)/BS);
        scale_kernel<<<GRID,BS,0,s>>>((float*)tdQ.data, n, scale);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("scale dQ", cudaGetLastError());
        }
      }
    }
  }

  // 9) dK = scale * (Q^T @ gS)^T
  if (dK){
    float* Qt_buf = nullptr;
    if (cudaMalloc(&Qt_buf, sizeof(float) * (size_t)D * (size_t)M) != cudaSuccess){
      cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
      return RTERR("cudaMalloc Qt_buf (dK path)");
    }

    for (int b=0; b<B; ++b){
      for (int h=0; h<H; ++h){
        ai::Tensor tQ { (char*)Q.data + ((size_t)((((size_t)b*H+h)*M)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,D}, {D,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tgS{ (char*)dgS    + ((size_t)((((size_t)b*H+h)*M)*N))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {M,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        ai::Tensor tdK{ (char*)dK->data + ((size_t)((((size_t)b*H+h)*N)*D))*sizeof(float),
                        {ai::DType::F32, ai::Layout::RowMajor, {N,D}, {D,1}}, ai::Device::CUDA, dK->device_index };

        // Q(M×D) -> Q^T(D×M)
        {
          dim3 blk(32,8);
          dim3 grd((D + blk.x - 1)/blk.x, (M + blk.y - 1)/blk.y);
          transpose_rm_f32<<<grd, blk, 0, s>>>((const float*)tQ.data, Qt_buf, /*R*/M, /*C*/D);
          if (cudaPeekAtLastError()!=cudaSuccess){
            cudaFree(Qt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
            return RTERR("transpose Q->Qt (dK path)", cudaGetLastError());
          }
        }
        ai::Tensor tQt{ Qt_buf, {ai::DType::F32, ai::Layout::RowMajor, {D,M}, {M,1}}, ai::Device::CUDA, Q.device_index };

        // T2 = Q^T(D×M) @ gS(M×N) = (D×N)
        float* dT2 = nullptr;
        if (cudaMalloc(&dT2, sizeof(float)*(size_t)D*(size_t)N)!=cudaSuccess){
          cudaFree(Qt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("cudaMalloc T2 (dK path)");
        }
        ai::Tensor tT2{ dT2, {ai::DType::F32, ai::Layout::RowMajor, {D,N}, {N,1}}, ai::Device::CUDA, Q.device_index };
        if (ai::ops::gemm_run(tQt, tgS, nullptr, tT2, g, stream)!=0){
          cudaFree(dT2); cudaFree(Qt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("gemm T2 = Q^T @ gS");
        }

        // dK = (T2)^T scaled
        dim3 blk2(32,8), grd2((N + blk2.x -1)/blk2.x, (D + blk2.y -1)/blk2.y);
        transpose_rm_f32<<<grd2,blk2,0,s>>>(dT2, (float*)tdK.data, /*R*/D, /*C*/N);
        cudaFree(dT2);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(Qt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("transpose tdK", cudaGetLastError());
        }

        long long n = (long long)N*(long long)D;
        int BS=256, GRID=(int)((n+BS-1)/BS);
        scale_kernel<<<GRID,BS,0,s>>>((float*)tdK.data, n, scale);
        if (cudaPeekAtLastError()!=cudaSuccess){
          cudaFree(Qt_buf); cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
          return RTERR("scale dK", cudaGetLastError());
        }
      }
    }
    cudaFree(Qt_buf);
  }

  // 최종 커널 에러 체크
  cudaError_t e = cudaPeekAtLastError();
  if (e != cudaSuccess) {
    cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
    return RTERR("final cudaPeekAtLastError", e);
  }

  cudaFree(dKt); cudaFree(dS); cudaFree(dP); cudaFree(dgP); cudaFree(dgS);
  return ai::Status::Ok;
}

} // namespace ai
