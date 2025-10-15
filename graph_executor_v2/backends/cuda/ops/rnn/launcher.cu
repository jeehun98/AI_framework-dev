#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cassert>

#include "backends/cuda/ops/gemm/api.hpp"   // GemmCudaLaunch (epilogue)
#include "backends/cuda/ops/rnn/api.hpp"    // this module's API

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/op_schema.hpp"
#endif

namespace ai {

// ========== 커널 런처 전방선언(Prototype) — cuh 없이 사용 ==========
void apply_act_rows_local_launcher(const float* Z, float* Y,
                                   int N, int H, int act_code, float slope,
                                   cudaStream_t s);
void apply_dact_rows_local_launcher(const float* gy_post, const float* Z, float* gy,
                                    int N, int H, int act_code, float slope,
                                    cudaStream_t s);
void add_dhnext_into_grows_launcher(const float* dXH, float* dG,
                                    int N, int I, int H, cudaStream_t s);
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s);
void add_transpose_into_launcher(float* dWcat, const float* Tmp_H_IpH,
                                 int IpH, int H, cudaStream_t s);
void reduce_db_rows_NH_launcher(const float* G, float* db, int N, int H, cudaStream_t s);

// ---- helpers ----
static inline bool is3_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==3;
}
static inline bool is2_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==2;
}
static inline bool is1_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// ======================= Forward =======================
Status RnnCudaLaunch(const Tensor& X,   // [N,T,I]
                     const Tensor& Wx,  // [I,H]
                     const Tensor& Wh,  // [H,H]
                     const Tensor* B,   // [H] (optional)
                     const Tensor& h0,  // [N,H]
                     Tensor& Y,         // [N,T,H]
                     const RnnAttrs& a,
                     StreamHandle stream,
                     Tensor* Z_saved,               // [N,T,H] optional
                     const RnnWorkspaceFwd* ws)     // workspace (no alloc)
{
  if (!is3_f32_cuda(X) || !is2_f32_cuda(Wx) || !is2_f32_cuda(Wh) || !is2_f32_cuda(h0)) return Status::Invalid;
  if (B && a.with_bias && !is1_f32_cuda(*B)) return Status::Invalid;
  if (!is3_f32_cuda(Y)) return Status::Invalid;

  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];

  const int64_t Iw = Wx.desc.shape[0];
  const int64_t H  = Wx.desc.shape[1];
  if (Iw != I) return Status::ShapeMismatch;
  if (Wh.desc.shape[0]!=H || Wh.desc.shape[1]!=H) return Status::ShapeMismatch;

  if (Y.desc.shape[0]!=N || Y.desc.shape[1]!=T || Y.desc.shape[2]!=H) return Status::ShapeMismatch;
  if (N<=0 || T<=0 || I<=0 || H<=0) return Status::Invalid;

  const bool want_z = a.save_z && (Z_saved!=nullptr);
  if (want_z) {
    if (!is3_f32_cuda(*Z_saved)) return Status::Invalid;
    if (Z_saved->desc.shape[0]!=N || Z_saved->desc.shape[1]!=T || Z_saved->desc.shape[2]!=H) return Status::ShapeMismatch;
  }

  if (!ws || !ws->XH_cat || !ws->Y_rows || !ws->W_cat || (want_z && !ws->Z_rows))
    return Status::MissingInput;

  auto s = to_cuda(stream);

  // W_cat = [Wx ; Wh]  (row-major: [I+H, H])
  cudaMemcpyAsync(ws->W_cat, Wx.data, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(ws->W_cat + (size_t)I*H, Wh.data, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);

  ai::GemmAttrs g{};
  g.act         = a.act;
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = (B && a.with_bias);
  g.save_z      = want_z;

  const ai::Tensor* BiasPtr = nullptr;
  ai::Tensor tBias{};
  if (g.with_bias) {
    tBias.data         = const_cast<float*>(static_cast<const float*>(B->data));
    tBias.device       = ai::Device::CUDA; tBias.device_index = 0;
    tBias.desc.dtype   = ai::DType::F32;
    tBias.desc.layout  = ai::Layout::RowMajor;
    tBias.desc.shape   = { (int64_t)H };
    tBias.desc.stride  = { 1 };
    BiasPtr = &tBias;
  }

  Tensor tA{}, tB{}, tO{}, tZcap{};
  const float* h_prev = static_cast<const float*>(h0.data);

  const size_t X_pitch_elems_dst = (size_t)(I + H);
  const size_t X_pitch_elems_src = (size_t)(T * I);
  const size_t H_pitch_elems     = (size_t)H;

  for (int64_t t=0; t<T; ++t) {
    // X[:,t,:] -> XH_cat[:, :I]
    {
      const void* src_base = static_cast<const float*>(X.data) + (size_t)t * I;
      void* dst_base = ws->XH_cat;
      cudaMemcpy2DAsync(dst_base, X_pitch_elems_dst * sizeof(float),
                        src_base, X_pitch_elems_src * sizeof(float),
                        I * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    // h_prev -> XH_cat[:, I:]
    {
      const void* src_h = h_prev;
      void* dst_h = ws->XH_cat + I;
      cudaMemcpy2DAsync(dst_h, X_pitch_elems_dst * sizeof(float),
                        src_h, H_pitch_elems * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    // GEMM: [N, I+H] @ [I+H, H] -> [N, H]
    tA = Tensor{ ws->XH_cat, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)(I+H)}, {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
    tB = Tensor{ ws->W_cat,  {DType::F32, Layout::RowMajor, {(int64_t)(I+H), (int64_t)H}, {(int64_t)H, 1}},   Device::CUDA, 0 };
    tO = Tensor{ ws->Y_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H},     {(int64_t)H, 1}},   Device::CUDA, 0 };
    if (want_z) {
      tZcap = Tensor{ ws->Z_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H}, {(int64_t)H, 1}}, Device::CUDA, 0 };
    }

    Status st = GemmCudaLaunch(tA, tB, BiasPtr, tO, g, stream, (want_z ? &tZcap : nullptr));
    if (st != Status::Ok) return st;

    // Y_rows → Y[:,t,:]
    {
      void* dst = static_cast<float*>(Y.data) + (size_t)t * H;
      const void* src = ws->Y_rows;
      cudaMemcpy2DAsync(dst, (size_t)(T*H) * sizeof(float),
                        src, (size_t)H * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    // Z_rows → Z[:,t,:]
    if (want_z) {
      void* dst = static_cast<float*>(Z_saved->data) + (size_t)t * H;
      const void* src = ws->Z_rows;
      cudaMemcpy2DAsync(dst, (size_t)(T*H) * sizeof(float),
                        src, (size_t)H * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    h_prev = ws->Y_rows;
  }

  return Status::Ok;
}

// ======================= Backward =======================
Status RnnCudaBackwardLaunch(const Tensor& X,       // [N,T,I]
                             const Tensor& Wx,      // [I,H]
                             const Tensor& Wh,      // [H,H]
                             const Tensor* B,       // [H] optional (unused)
                             const Tensor& h0,      // [N,H]
                             const Tensor& dY_post, // [N,T,H]
                             const Tensor& Z,       // [N,T,H]
                             Tensor* dWx,           // [I,H]
                             Tensor* dWh,           // [H,H]
                             Tensor* dB,            // [H]
                             Tensor* dh0,           // [N,H]
                             Tensor* dX,            // [N,T,I]
                             const RnnAttrs& a,
                             StreamHandle stream,
                             const RnnWorkspaceBwd* ws)
{
  if (!is3_f32_cuda(X) || !is2_f32_cuda(Wx) || !is2_f32_cuda(Wh) || !is2_f32_cuda(h0)) return Status::Invalid;
  if (!is3_f32_cuda(dY_post) || !is3_f32_cuda(Z)) return Status::Invalid;
  if (dWx && (!is2_f32_cuda(*dWx) || dWx->desc.shape[0]!=X.desc.shape[2] || dWx->desc.shape[1]!=Wx.desc.shape[1])) return Status::ShapeMismatch;
  if (dWh && (!is2_f32_cuda(*dWh) || dWh->desc.shape[0]!=Wx.desc.shape[1] || dWh->desc.shape[1]!=Wx.desc.shape[1])) return Status::ShapeMismatch;
  if (dB  && a.with_bias && (!is1_f32_cuda(*dB) || dB->desc.shape[0]!=Wx.desc.shape[1])) return Status::ShapeMismatch;
  if (dh0 && (!is2_f32_cuda(*dh0) || dh0->desc.shape[0]!=X.desc.shape[0] || dh0->desc.shape[1]!=Wx.desc.shape[1])) return Status::ShapeMismatch;
  if (dX  && (!is3_f32_cuda(*dX)  || dX->desc.shape[0]!=X.desc.shape[0] || dX->desc.shape[1]!=X.desc.shape[1] || dX->desc.shape[2]!=X.desc.shape[2])) return Status::ShapeMismatch;

  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];
  const int64_t H = Wx.desc.shape[1];
  if (N<=0 || T<=0 || I<=0 || H<=0) return Status::Invalid;

  if (!ws || !ws->XH_cat || !ws->G_rows || !ws->Z_rows || !ws->W_cat || !ws->dXH_cat || !ws->dWcat || !ws->TmpW)
    return Status::MissingInput;

  auto s = to_cuda(stream);

  // zero grads
  if (dWx) cudaMemsetAsync(dWx->data, 0, sizeof(float)*(size_t)I*H, s);
  if (dWh) cudaMemsetAsync(dWh->data, 0, sizeof(float)*(size_t)H*H, s);
  if (dB  && a.with_bias) cudaMemsetAsync(dB->data, 0, sizeof(float)*(size_t)H, s);
  if (dh0) cudaMemsetAsync(dh0->data, 0, sizeof(float)*(size_t)N*H, s);
  if (dX)  cudaMemsetAsync(dX->data, 0, sizeof(float)*(size_t)N*T*I, s);
  cudaMemsetAsync(ws->dWcat,  0, sizeof(float)*(size_t)(I+H)*H, s);
  cudaMemsetAsync(ws->dXH_cat,0, sizeof(float)*(size_t)N*(I+H), s);

  // W_cat = [Wx ; Wh] : [I+H, H]
  cudaMemcpyAsync(ws->W_cat, Wx.data, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(ws->W_cat + (size_t)I*H, Wh.data, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);

  ai::GemmAttrs g{}; g.act = ai::ActKind::None; g.with_bias=false;

  // pitches
  const size_t X_pitch_elems_dst = (size_t)(I + H);
  const size_t X_pitch_elems_src = (size_t)(T * I);
  const size_t H_pitch_elems     = (size_t)H;
  const size_t TH_pitch_elems    = (size_t)(T * H);

  for (int64_t t = T-1; t >= 0; --t) {
    // XH_cat[:, :I] = X[:,t,:]
    {
      const void* src_base = static_cast<const float*>(X.data) + (size_t)t * I;
      void* dst_base = ws->XH_cat;
      cudaMemcpy2DAsync(dst_base, X_pitch_elems_dst * sizeof(float),
                        src_base, X_pitch_elems_src * sizeof(float),
                        I * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    // XH_cat[:, I:] = (t==0 ? h0 : act(Z[:,t-1,:]))
    if (t == 0) {
      const void* src_h0 = h0.data;
      void* dst_h = ws->XH_cat + I;
      cudaMemcpy2DAsync(dst_h, X_pitch_elems_dst * sizeof(float),
                        src_h0, H_pitch_elems * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    } else {
      // Z_rows <- Z[:,t-1,:]
      {
        const void* src_z = static_cast<const float*>(Z.data) + (size_t)(t-1) * H;
        void* dst_z = ws->Z_rows;
        cudaMemcpy2DAsync(dst_z, H_pitch_elems * sizeof(float),
                          src_z, TH_pitch_elems * sizeof(float),
                          H * sizeof(float), (size_t)N,
                          cudaMemcpyDeviceToDevice, s);
      }
      apply_act_rows_local_launcher(ws->Z_rows, ws->Z_rows, (int)N, (int)H, (int)a.act, a.leaky_slope, s);
      // copy to tail
      {
        const void* src = ws->Z_rows;
        void* dst = ws->XH_cat + I;
        cudaMemcpy2DAsync(dst, X_pitch_elems_dst * sizeof(float),
                          src, H_pitch_elems * sizeof(float),
                          H * sizeof(float), (size_t)N,
                          cudaMemcpyDeviceToDevice, s);
      }
    }

    // G_rows = dY_post[:,t,:]
    {
      const void* src_gy = static_cast<const float*>(dY_post.data) + (size_t)t * H;
      void* dst_g = ws->G_rows; // [N,H]
      cudaMemcpy2DAsync(dst_g, H_pitch_elems * sizeof(float),
                        src_gy, TH_pitch_elems * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    // + dh_next (from previous iteration) => dXH_cat[:, I:]
    add_dhnext_into_grows_launcher(ws->dXH_cat, ws->G_rows, (int)N, (int)I, (int)H, s);

    // dact: use Z[:,t,:]
    {
      const void* src_z = static_cast<const float*>(Z.data) + (size_t)t * H;
      void* dst_z = ws->Z_rows; // reuse as [N,H]
      cudaMemcpy2DAsync(dst_z, H_pitch_elems * sizeof(float),
                        src_z, TH_pitch_elems * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    apply_dact_rows_local_launcher(ws->G_rows, ws->Z_rows, ws->G_rows, (int)N, (int)H, (int)a.act, a.leaky_slope, s);
    if (dB && a.with_bias) {
      reduce_db_rows_NH_launcher(ws->G_rows, static_cast<float*>(dB->data), (int)N, (int)H, s);
    }

    // (1) dXH = G_rows[N,H] @ (W_cat^T)[H,I+H]
    transpose_kernel_launcher(ws->W_cat, ws->TmpW, /*M=*/(int)(I+H), /*N=*/(int)H, s); // TmpW := W_cat^T (H, I+H)
    {
      Tensor tG { ws->G_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H},       {(int64_t)H, 1}},     Device::CUDA, 0 };
      Tensor tWT{ ws->TmpW,   {DType::F32, Layout::RowMajor, {(int64_t)H, (int64_t)(I+H)},   {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
      Tensor tO { ws->dXH_cat,{DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)(I+H)},   {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
      Status st = GemmCudaLaunch(tG, tWT, /*Bias*/nullptr, tO, g, stream, nullptr);
      if (st != Status::Ok) return st;
    }
    // scatter dX[:,t,:] = dXH[:, :I]
    if (dX){
      void* dst = static_cast<float*>(dX->data) + (size_t)t * I;
      const void* src = ws->dXH_cat; // [:, :I]
      cudaMemcpy2DAsync(dst, (size_t)(T*I) * sizeof(float),
                        src, (size_t)(I+H) * sizeof(float),
                        I * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    // (2) dWcat += (XH_cat^T @ G_rows) => Tmp := (G_rows^T @ XH_cat) [H, I+H], dWcat += Tmp^T
    transpose_kernel_launcher(ws->G_rows, ws->Z_rows, /*M=*/(int)N, /*N=*/(int)H, s); // Z_rows := G^T [H,N]
    {
      Tensor tGT{ ws->Z_rows, {DType::F32, Layout::RowMajor, {(int64_t)H, (int64_t)N},     {(int64_t)N, 1}},       Device::CUDA, 0 };
      Tensor tX { ws->XH_cat, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)(I+H)}, {(int64_t)(I+H), 1}},   Device::CUDA, 0 };
      Tensor tS { ws->TmpW,   {DType::F32, Layout::RowMajor, {(int64_t)H, (int64_t)(I+H)}, {(int64_t)(I+H), 1}},   Device::CUDA, 0 };
      Status st = GemmCudaLaunch(tGT, tX, /*Bias*/nullptr, tS, g, stream, nullptr); // S = G^T @ X
      if (st != Status::Ok) return st;
    }
    add_transpose_into_launcher(ws->dWcat, ws->TmpW, (int)(I+H), (int)H, s);
  } // for t

  // split dWcat -> dWx, dWh
  if (dWx){
    cudaMemcpyAsync(dWx->data, ws->dWcat, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  }
  if (dWh){
    cudaMemcpyAsync(dWh->data, ws->dWcat + (size_t)I*H, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  }

  // dh0 = dh_next (after t==0) = dXH_cat[:, I:]
  if (dh0){
    void* dst = dh0->data;             // [N,H]
    const void* src = ws->dXH_cat + I; // [:, I:]
    cudaMemcpy2DAsync(dst, (size_t)H * sizeof(float),
                      src, (size_t)(I+H) * sizeof(float),
                      H * sizeof(float), (size_t)N,
                      cudaMemcpyDeviceToDevice, s);
  }

  return Status::Ok;
}

} // namespace ai
