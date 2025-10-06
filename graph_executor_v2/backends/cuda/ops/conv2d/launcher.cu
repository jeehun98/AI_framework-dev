#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cassert>

#include "backends/cuda/ops/conv2d/api.hpp"
#include "backends/cuda/ops/gemm/api.hpp"   // GemmCudaLaunch (epilogue 사용)
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif


namespace ai {

// ===== utils & externs (기존 그대로) =====
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

// im2col/col2im/transpose 런처 (선언부)
void im2col_kernel_launcher(const float*, float*,
                            int,int,int, int,int, int,int, int,int, int,int, int,int, cudaStream_t);
void col2im_kernel_launcher(const float*, float*,
                            int,int,int, int,int, int,int, int,int, int,int, int,int, cudaStream_t);
// row-major transpose: in[M,N] -> out[N,M]
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t);

// ===== dB reduce (co-by-row) =====
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

// ===== pack/unpack W (기존 그대로) =====
__global__ void pack_w_oihw_to_KC(const float* __restrict__ W, float* __restrict__ out_KC,
                                  int Cout, int Cin, int Kh, int Kw);
__global__ void pack_w_oihw_to_CK(const float* __restrict__ W, float* __restrict__ out_CK,
                                  int Cout, int Cin, int Kh, int Kw);
__global__ void unpack_ck_to_oihw_add(const float* __restrict__ dWpack, float* __restrict__ dW,
                                      int Cout, int Cin, int Kh, int Kw);

// ===== activation backward (elementwise) on [Cout, HWo] =====
__device__ __forceinline__ float dact(ai::ActKind act, float z, float gy, float slope) {
  switch (act) {
    case ai::ActKind::None:    return gy;
    case ai::ActKind::ReLU:    return (z > 0.f) ? gy : 0.f;
    case ai::ActKind::LeakyReLU:return (z > 0.f) ? gy : slope * gy;
    case ai::ActKind::Sigmoid: {
      float s = 1.f / (1.f + __expf(-z));
      return gy * s * (1.f - s);
    }
    case ai::ActKind::Tanh: {
      float t = tanhf(z);
      return gy * (1.f - t*t);
    }
    case ai::ActKind::GELU: {
      // fast approx derivative
      const float c = sqrtf(2.f / 3.1415926535f);
      float z3 = z*z*z;
      float th = tanhf(c*(z + 0.044715f*z3));
      float dtanh = (1 - th*th) * c * (1 + 0.134145f*z*z);
      return gy * (0.5f*(1 + th) + 0.5f*z*dtanh);
    }
    default: return gy;
  }
}

// gy[Cout,HWo] = gy_post[Cout,HWo] ⊙ act'(Z[Cout,HWo])
__global__ void apply_dact_rows(const float* __restrict__ gy_post,
                                const float* __restrict__ Z_rows,
                                float* __restrict__ gy_rows,
                                int Cout, int HWo, ai::ActKind act, float slope)
{
  int hw = blockIdx.x * blockDim.x + threadIdx.x;
  int co = blockIdx.y * blockDim.y + threadIdx.y;
  if (co < Cout && hw < HWo) {
    size_t idx = (size_t)co * HWo + hw;
    gy_rows[idx] = dact(act, Z_rows[idx], gy_post[idx], slope);
  }
}

// ======================= Forward =======================
Status Conv2DCudaLaunch(const Tensor& X, const Tensor& W, const Tensor* B, Tensor& Y,
                        const Conv2DAttrs& a, StreamHandle stream, Tensor* Z_saved)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(Y)) return Status::Invalid;
  if (!is4_f32_cuda(W)) return Status::Invalid;
  if (a.groups != 1)    return Status::Unimplemented;

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

  // Z_saved 체크
  const bool want_z = a.save_z && (Z_saved != nullptr);
  if (want_z) {
    if (!is4_f32_cuda(*Z_saved)) return Status::Invalid;
    if (Z_saved->desc.shape[0]!=N || Z_saved->desc.shape[1]!=Cout ||
        Z_saved->desc.shape[2]!=Ho || Z_saved->desc.shape[3]!=Wo) return Status::ShapeMismatch;
  }

  const float* dW = static_cast<const float*>(W.data);
  const float* dB = (B && B->data && a.with_bias) ? static_cast<const float*>(B->data) : nullptr;

  const int K   = Cin*Kh*Kw;
  const int HWo = Ho*Wo;
  auto s = to_cuda(stream);

  // workspaces (임시 유지 — CUDA Graph화 시 외부 주입/캐시로 교체)
  float *dCol=nullptr, *W_KC=nullptr, *Y_tmp=nullptr, *Z_tmp=nullptr;
  if (cudaMalloc(&dCol,  sizeof(float)*HWo*K)    != cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&W_KC,  sizeof(float)*K*Cout)   != cudaSuccess) { cudaFree(dCol); return Status::RuntimeError; }
  if (cudaMalloc(&Y_tmp, sizeof(float)*HWo*Cout) != cudaSuccess) { cudaFree(dCol); cudaFree(W_KC); return Status::RuntimeError; }
  if (want_z) {
    if (cudaMalloc(&Z_tmp, sizeof(float)*HWo*Cout) != cudaSuccess) {
      cudaFree(dCol); cudaFree(W_KC); cudaFree(Y_tmp);
      return Status::RuntimeError;
    }
  }

  // pack W → [K, Cout]
  {
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_KC<<<grid, block, 0, s>>>(dW, W_KC, Cout, Cin, Kh, Kw);
  }

  // GEMM attrs: epilogue 사용
  ai::GemmAttrs g{};
  g.act         = a.act;
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = (dB != nullptr);
  g.save_z      = want_z; // regemm에게 pre-activation 저장 요청

  for (int n=0; n<N; ++n) {
    const float* x_n = static_cast<const float*>(X.data) + (size_t)n*Cin*H*Wd;
    float*       y_n = static_cast<float*>(Y.data)       + (size_t)n*Cout*Ho*Wo;
    float*       z_n = want_z ? static_cast<float*>(Z_saved->data) + (size_t)n*Cout*Ho*Wo : nullptr;

    // im2col: [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd, Kh, Kw,
      a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
      Ho, Wo, s
    );

    // GEMM: [HWo,K] @ [K,Cout] -> [HWo,Cout] into Y_tmp (epilogue bias+act). Z_tmp는 pre-act 저장.
    Tensor tA{dCol,  {DType::F32, Layout::RowMajor, {HWo, K},    {K, 1}},     Device::CUDA, 0};
    Tensor tB{W_KC,  {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}},  Device::CUDA, 0};
    Tensor tY{Y_tmp, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}},  Device::CUDA, 0};

    Tensor tZcap{};
    if (want_z) {
      tZcap = Tensor{Z_tmp, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout,1}}, Device::CUDA, 0};
    }
    
    const ai::Tensor* BiasPtr = nullptr;
    ai::Tensor BiasT;  // 로컬 lvalue로 잡아둔다
    if (dB) {
      BiasT.data         = const_cast<float*>(dB);
      BiasT.device       = ai::Device::CUDA;
      BiasT.device_index = 0;
      BiasT.desc.dtype   = ai::DType::F32;
      BiasT.desc.layout  = ai::Layout::RowMajor;
      BiasT.desc.shape   = { (int64_t)Cout };
      BiasT.desc.stride  = { 1 };
      BiasPtr = &BiasT;
    }

    ai::Status st = ai::GemmCudaLaunch(
        tA, tB,
        BiasPtr,      // ← 임시 주소 대신 로컬 변수 주소
        tY, g, stream,
        (want_z ? &tZcap : nullptr)
    );

    if (st != ai::Status::Ok) {
      cudaFree(dCol); cudaFree(W_KC); cudaFree(Y_tmp); if (Z_tmp) cudaFree(Z_tmp);
      return st;
    }

    // transpose: [HWo, Cout] -> [Cout, HWo] directly into y_n (NCHW contiguous)
    transpose_kernel_launcher(Y_tmp, y_n, /*M=*/HWo, /*N=*/Cout, s);

    if (want_z) {
      // Z도 동일하게 NCHW로 저장
      transpose_kernel_launcher(Z_tmp, z_n, /*M=*/HWo, /*N=*/Cout, s);
    }
  }

  cudaFree(dCol);
  cudaFree(W_KC);
  cudaFree(Y_tmp);
  if (Z_tmp) cudaFree(Z_tmp);
  return Status::Ok;
}

// ======================= Backward =======================
Status Conv2DCudaBackwardLaunch(const Tensor& X, const Tensor& W, const Tensor& dY_post,
                                const Tensor& Z, Tensor* dW, Tensor* dB, Tensor* dX,
                                const Conv2DAttrs& a, StreamHandle stream)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(W) || !is4_f32_cuda(dY_post) || !is4_f32_cuda(Z)) return Status::Invalid;
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

  if (dY_post.desc.shape[0]!=N || dY_post.desc.shape[1]!=Cout || dY_post.desc.shape[2]!=Ho || dY_post.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;
  if (Z.desc.shape[0]!=N || Z.desc.shape[1]!=Cout || Z.desc.shape[2]!=Ho || Z.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;

  if (dW) {
    if (!is4_f32_cuda(*dW) ||
        dW->desc.shape[0]!=Cout || dW->desc.shape[1]!=Cin ||
        dW->desc.shape[2]!=Kh   || dW->desc.shape[3]!=Kw) return Status::ShapeMismatch;
  }
  if (dB) { if (!(is1_f32_cuda(*dB) && (int)dB->desc.shape[0]==Cout)) return Status::ShapeMismatch; }
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
  float *gy_rows=nullptr, *Z_rows=nullptr; // [Cout,HWo]
  if (cudaMalloc(&dCol, sizeof(float)*HWo*K) != cudaSuccess) return Status::RuntimeError;

  size_t tmp_elems = (size_t)std::max(Cout*K, HWo*K);
  if (cudaMalloc(&dTmp, sizeof(float)*tmp_elems) != cudaSuccess) { cudaFree(dCol); return Status::RuntimeError; }

  if (dB) cudaMemsetAsync(dB->data, 0, sizeof(float)*Cout, s);
  if (dW) cudaMemsetAsync(dW->data, 0, sizeof(float)*Cout*Cin*Kh*Kw, s);
  if (dX) cudaMemsetAsync(dX->data, 0, sizeof(float)*N*Cin*H*Wd, s);

  // dX용 W_CK & dY_HT 준비
  if (dX) {
    const float* dWsrc = static_cast<const float*>(W.data);
    if (cudaMalloc(&W_CK, sizeof(float)*Cout*K) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp);
      return Status::RuntimeError;
    }
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_CK<<<grid, block, 0, s>>>(dWsrc, W_CK, Cout, Cin, Kh, Kw);

    if (cudaMalloc(&dY_HT, sizeof(float)*HWo*Cout) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp); cudaFree(W_CK);
      return Status::RuntimeError;
    }
  }

  // gy_rows & Z_rows: [Cout,HWo] (act bwd를 위해 NCHW -> rows로 전치)
  if (cudaMalloc(&gy_rows, sizeof(float)*Cout*HWo) != cudaSuccess) {
    cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT);
    return Status::RuntimeError;
  }
  if (cudaMalloc(&Z_rows, sizeof(float)*Cout*HWo) != cudaSuccess) {
    cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT); cudaFree(gy_rows);
    return Status::RuntimeError;
  }

  // dW 누적 버퍼: [Cout, K]
  if (dW) {
    if (cudaMalloc(&dWpack, sizeof(float)*Cout*K) != cudaSuccess) {
      cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT);
      cudaFree(gy_rows); cudaFree(Z_rows);
      return Status::RuntimeError;
    }
    cudaMemsetAsync(dWpack, 0, sizeof(float)*Cout*K, s);
  }

  // GEMM attrs (No epilogue here; epilogue는 forward에서만)
  ai::GemmAttrs g{}; g.act=ai::ActKind::None; g.with_bias=false;

  for (int n=0; n<N; ++n) {
    const float* x_n   = static_cast<const float*>(X.data)      + (size_t)n*Cin*H*Wd;
    const float* gy_nP = static_cast<const float*>(dY_post.data)+ (size_t)n*Cout*Ho*Wo; // post-act grad
    const float* z_n   = static_cast<const float*>(Z.data)      + (size_t)n*Cout*Ho*Wo;

    // NCHW → [Cout,HWo]
    transpose_kernel_launcher(gy_nP, gy_rows, /*M=*/Ho*Wo, /*N=*/Cout, s); // [HWo,Cout]->[Cout,HWo]
    transpose_kernel_launcher(z_n,   Z_rows,  /*M=*/Ho*Wo, /*N=*/Cout, s);

    // gy_rows = gy_post ⊙ act'(Z_rows)
    {
      dim3 block(128, 1), grid((HWo + block.x - 1)/block.x, Cout);
      apply_dact_rows<<<grid, block, 0, s>>>(gy_rows, Z_rows, gy_rows, Cout, HWo, a.act, a.leaky_slope);
    }

    // im2col(X[n]) → dCol [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd, Kh, Kw,
      a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
      Ho, Wo, s
    );

    // dB: sum over HWo (gy_rows is [Cout, HWo])
    if (dB && a.with_bias) {
      reduce_db_rows_kernel_launcher(gy_rows, static_cast<float*>(dB->data), Cout, HWo, s);
    }

    // dW: dWpack += gy_rows[Cout,HWo] @ X_col[HWo, K] -> [Cout, K]
    if (dW) {
      Tensor tA{const_cast<float*>(gy_rows), {DType::F32, Layout::RowMajor, {Cout, HWo}, {HWo, 1}}, Device::CUDA, 0};
      Tensor tB{dCol,                         {DType::F32, Layout::RowMajor, {HWo,  K},   {K,   1}}, Device::CUDA, 0};
      Tensor tO{dTmp,                         {DType::F32, Layout::RowMajor, {Cout, K},   {K,   1}}, Device::CUDA, 0};
      ai::Status st = ai::GemmCudaLaunch(tA, tB, /*Bias*/nullptr, tO, g, stream, nullptr);
      if (st != ai::Status::Ok) { /* free & return */ 
        cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT);
        if (dWpack) cudaFree(dWpack); cudaFree(gy_rows); cudaFree(Z_rows);
        return st;
      }
      int total = Cout * K;
      dim3 b(256), gr((total + 255)/256);
      kadd_kernel<<<gr, b, 0, s>>>(dWpack, dTmp, total);
    }

    // dX: dX += ( [HWo,Cout] @ [Cout,K] ) -> [HWo,K] -> col2im
    if (dX) {
      transpose_kernel_launcher(gy_rows, dY_HT, /*M=*/Cout, /*N=*/HWo, s); // [Cout,HWo]->[HWo,Cout]
      Tensor tA{dY_HT, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};
      Tensor tB{W_CK,  {DType::F32, Layout::RowMajor, {Cout, K},   {K,    1}}, Device::CUDA, 0};
      Tensor tO{dTmp,  {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0};
      ai::Status st = ai::GemmCudaLaunch(tA, tB, /*Bias*/nullptr, tO, g, stream, nullptr);
      if (st != ai::Status::Ok) { /* free & return */
        cudaFree(dCol); cudaFree(dTmp); if (W_CK) cudaFree(W_CK); if (dY_HT) cudaFree(dY_HT);
        if (dWpack) cudaFree(dWpack); cudaFree(gy_rows); cudaFree(Z_rows);
        return st;
      }
      float* dx_n = static_cast<float*>(dX->data) + (size_t)n*Cin*H*Wd;
      col2im_kernel_launcher(
        dTmp, dx_n,
        Cin, H, Wd, Kh, Kw,
        a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
        Ho, Wo, s
      );
    }
  }

  // dWpack[Cout,K] -> dW[O,I,H,W]
  if (dW) {
    const int K = Cin*Kh*Kw;
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    unpack_ck_to_oihw_add<<<grid, block, 0, s>>>(dWpack, static_cast<float*>(dW->data), Cout, Cin, Kh, Kw);
  }

  cudaFree(dCol);
  cudaFree(dTmp);
  if (W_CK)   cudaFree(W_CK);
  if (dWpack) cudaFree(dWpack);
  if (dY_HT)  cudaFree(dY_HT);
  if (gy_rows)cudaFree(gy_rows);
  if (Z_rows) cudaFree(Z_rows);
  return Status::Ok;
}

} // namespace ai
