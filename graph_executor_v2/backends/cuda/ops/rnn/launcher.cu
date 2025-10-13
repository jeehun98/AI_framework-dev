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

  const bool want_z = a.save_z && (Z_saved!=nullptr);
  if (want_z) {
    if (!is3_f32_cuda(*Z_saved)) return Status::Invalid;
    if (Z_saved->desc.shape[0]!=N || Z_saved->desc.shape[1]!=T || Z_saved->desc.shape[2]!=H) return Status::ShapeMismatch;
  }

  if (!ws || !ws->XH_cat || !ws->Y_rows || !ws->W_cat || (want_z && !ws->Z_rows))
    return Status::MissingInput;

  auto s = to_cuda(stream);

  // Pack Wcat = [Wx ; Wh] (D2D 2회 or 커널)
  // memcpy 2회:
  cudaMemcpyAsync(ws->W_cat, Wx.data, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(ws->W_cat + (size_t)I*H, Wh.data, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  // (대신 pack_wcat_from_wx_wh_launcher(Wx.data, Wh.data, ws->W_cat, I, H, s); 도 가능)

  // GEMM epilogue 설정
  ai::GemmAttrs g{};
  g.act         = a.act;
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = (B && a.with_bias);
  g.save_z      = want_z;

  // Bias 텐서 래핑 (optional)
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

  // 텐서 래핑 (시점마다 재사용)
  Tensor tA{}, tB{}, tO{}, tZcap{};

  // 초기 h_prev = h0
  const float* h_prev = static_cast<const float*>(h0.data);

  // 시점 루프
  for (int64_t t=0; t<T; ++t) {
    // X_t: [N,I]가 물리적으로 연속이 아님 → batch별 D2D로 XH_cat 구성
    for (int64_t n=0; n<N; ++n) {
      const float* xn = static_cast<const float*>(X.data) + (size_t)((n*T + t)*I);
      float* xh_n = ws->XH_cat + (size_t)n*(I+H);
      cudaMemcpyAsync(xh_n, xn, (size_t)I*sizeof(float), cudaMemcpyDeviceToDevice, s);
      const float* hp = h_prev + (size_t)n*H;
      cudaMemcpyAsync(xh_n + I, hp, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
    }

    // GEMM: [N, I+H] @ [I+H, H] -> [N, H], epilogue(activation,bias,save_z)
    tA = Tensor{ ws->XH_cat, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)(I+H)}, {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
    tB = Tensor{ ws->W_cat,  {DType::F32, Layout::RowMajor, {(int64_t)(I+H), (int64_t)H}, {(int64_t)H, 1}},   Device::CUDA, 0 };
    tO = Tensor{ ws->Y_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H}, {(int64_t)H, 1}},       Device::CUDA, 0 };
    if (want_z) {
      tZcap = Tensor{ ws->Z_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H}, {(int64_t)H, 1}}, Device::CUDA, 0 };
    }

    Status st = GemmCudaLaunch(tA, tB, BiasPtr, tO, g, stream, (want_z ? &tZcap : nullptr));
    if (st != Status::Ok) return st;

    // Y[:,t,:] <- Y_rows (그리고 Z 저장)
    for (int64_t n=0; n<N; ++n) {
      float* y_nt = static_cast<float*>(Y.data) + (size_t)((n*T + t)*H);
      const float* srcY = ws->Y_rows + (size_t)n*H;
      cudaMemcpyAsync(y_nt, srcY, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
      if (want_z) {
        float* z_nt = static_cast<float*>(Z_saved->data) + (size_t)((n*T + t)*H);
        const float* srcZ = ws->Z_rows + (size_t)n*H;
        cudaMemcpyAsync(z_nt, srcZ, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
      }
    }

    // 다음 시점의 h_prev = 현재 출력(Y_rows)
    h_prev = ws->Y_rows;
  }

  return Status::Ok;
}

// ======================= Backward =======================
Status RnnCudaBackwardLaunch(const Tensor& X,       // [N,T,I]
                             const Tensor& Wx,      // [I,H]
                             const Tensor& Wh,      // [H,H]
                             const Tensor* B,       // [H] optional
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

  if (!ws || !ws->XH_cat || !ws->G_rows || !ws->Z_rows || !ws->W_cat || !ws->dXH_cat || !ws->dWcat || !ws->TmpW)
    return Status::MissingInput;

  auto s = to_cuda(stream);

  // 초기화 (안전)
  if (dWx) cudaMemsetAsync(dWx->data, 0, sizeof(float)*(size_t)I*H, s);
  if (dWh) cudaMemsetAsync(dWh->data, 0, sizeof(float)*(size_t)H*H, s);
  if (dB  && a.with_bias) cudaMemsetAsync(dB->data, 0, sizeof(float)*(size_t)H, s);
  if (dh0) cudaMemsetAsync(dh0->data, 0, sizeof(float)*(size_t)N*H, s);
  if (dX)  cudaMemsetAsync(dX->data, 0, sizeof(float)*(size_t)N*T*I, s);

  cudaMemsetAsync(ws->dWcat, 0, sizeof(float)*(size_t)(I+H)*H, s);
  cudaMemsetAsync(ws->dXH_cat, 0, sizeof(float)*(size_t)N*(I+H), s);

  // Wcat = [Wx ; Wh]
  cudaMemcpyAsync(ws->W_cat, Wx.data, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(ws->W_cat + (size_t)I*H, Wh.data, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  // (필요 시 pack_wcat_from_wx_wh_launcher(...) 사용 가능)

  // GEMM attrs (역전파에서는 epilogue 미사용)
  ai::GemmAttrs g{}; g.act = ai::ActKind::None; g.with_bias=false;

  // dh_next = 0 (ws->dXH_cat[:, I:] 영역 재사용)
  // 루프: t = T-1 .. 0

  for (int64_t t = T-1; t >= 0; --t) {

    // --- (A) XH_cat = [X_t | h_{t-1}] ---
    for (int64_t n=0; n<N; ++n) {
      const float* xn = static_cast<const float*>(X.data) + (size_t)((n*T + t)*I);
      float* xh_n = ws->XH_cat + (size_t)n*(I+H);
      cudaMemcpyAsync(xh_n, xn, (size_t)I*sizeof(float), cudaMemcpyDeviceToDevice, s);
    }

    if (t == 0) {
      // h_{-1} = h0
      for (int64_t n=0; n<N; ++n) {
        const float* h0n = static_cast<const float*>(h0.data) + (size_t)n*H;
        float* xh_n = ws->XH_cat + (size_t)n*(I+H);
        cudaMemcpyAsync(xh_n + I, h0n, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
      }
    } else {
      // h_{t-1} = act(Z[:,t-1,:])
      // 1) ws->Z_rows <- Z[:,t-1,:]
      for (int64_t n=0; n<N; ++n) {
        const float* z_prev = static_cast<const float*>(Z.data) + (size_t)((n*T + (t-1))*H);
        float* zbuf = ws->Z_rows + (size_t)n*H;
        cudaMemcpyAsync(zbuf, z_prev, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
      }
      // 2) ws->Z_rows <- act(ws->Z_rows)
      apply_act_rows_launcher(ws->Z_rows, ws->Z_rows, (int)N, (int)H, (int)a.act, a.leaky_slope, s);
      // 3) XH_cat[:, I:] <- ws->Z_rows
      for (int64_t n=0; n<N; ++n) {
        const float* src = ws->Z_rows + (size_t)n*H;
        float* xh_n = ws->XH_cat + (size_t)n*(I+H);
        cudaMemcpyAsync(xh_n + I, src, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
      }
    }

    // --- (B) G_rows 준비: dY_post + dh_next ---
    for (int64_t n=0; n<N; ++n) {
      const float* gy_nt = static_cast<const float*>(dY_post.data) + (size_t)((n*T + t)*H);
      float* grow = ws->G_rows + (size_t)n*H;
      cudaMemcpyAsync(grow, gy_nt, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    add_rows_strided_launcher(ws->G_rows, ws->dXH_cat, (int)N, (int)H, (int)(I+H), (int)I, s);

    // --- (C) 비선형 미분: 현재 시점 Z[:,t,:] 사용 ---
    // ws->Z_rows <- Z[:,t,:]
    for (int64_t n=0; n<N; ++n) {
      const float* z_t = static_cast<const float*>(Z.data) + (size_t)((n*T + t)*H);
      float* zbuf = ws->Z_rows + (size_t)n*H;
      cudaMemcpyAsync(zbuf, z_t, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    apply_dact_rows_launcher(ws->G_rows, ws->Z_rows, ws->G_rows, (int)N, (int)H, (int)a.act, a.leaky_slope, s);

    // --- (D) dB, dWcat, dXH, dX 업데이트 (이후 로직 동일) ---
    if (dB && a.with_bias) {
      reduce_db_rows_kernel_launcher(ws->G_rows, static_cast<float*>(dB->data), (int)N, (int)H, s);
    }

    // dWcat_t = (XH_cat)^T @ G_rows -> TmpW; dWcat += TmpW
    {
      Tensor tXt{ ws->XH_cat, {DType::F32, Layout::RowMajor, {(int64_t)(I+H), (int64_t)N}, {(int64_t)N, 1}}, Device::CUDA, 0 };
      Tensor tG { ws->G_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H},    {(int64_t)H, 1}}, Device::CUDA, 0 };
      Tensor tO { ws->TmpW,   {DType::F32, Layout::RowMajor, {(int64_t)(I+H), (int64_t)H},{(int64_t)H, 1}}, Device::CUDA, 0 };
      Status st = GemmCudaLaunch(tXt, tG, /*Bias*/nullptr, tO, ai::GemmAttrs{}, stream, nullptr);
      if (st != Status::Ok) return st;
      kadd_vec_launcher(ws->dWcat, ws->TmpW, (int)((I+H)*H), s);
    }

    // dXH = G_rows @ (Wcat)^T  → [N, I+H]
    {
      Tensor tG { ws->G_rows, {DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)H},       {(int64_t)H, 1}},     Device::CUDA, 0 };
      Tensor tWT{ ws->W_cat,  {DType::F32, Layout::RowMajor, {(int64_t)H, (int64_t)(I+H)},   {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
      Tensor tO { ws->dXH_cat,{DType::F32, Layout::RowMajor, {(int64_t)N, (int64_t)(I+H)},   {(int64_t)(I+H), 1}}, Device::CUDA, 0 };
      Status st = GemmCudaLaunch(tG, tWT, /*Bias*/nullptr, tO, ai::GemmAttrs{}, stream, nullptr);
      if (st != Status::Ok) return st;
    }

    if (dX){
      for (int64_t n=0; n<N; ++n) {
        float* dx_nt = static_cast<float*>(dX->data) + (size_t)((n*T + t)*I);
        const float* src = ws->dXH_cat + (size_t)n*(I+H);
        cudaMemcpyAsync(dx_nt, src, (size_t)I*sizeof(float), cudaMemcpyDeviceToDevice, s);
      }
    }
  } // for t


  // dWcat → dWx, dWh
  if (dWx){
    cudaMemcpyAsync(dWx->data, ws->dWcat, (size_t)I*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  }
  if (dWh){
    cudaMemcpyAsync(dWh->data, ws->dWcat + (size_t)I*H, (size_t)H*H*sizeof(float), cudaMemcpyDeviceToDevice, s);
  }

  // dh0 = dh_next (t = -1) = 현재 ws->dXH_cat[:, I:]
  if (dh0){
    for (int64_t n=0; n<N; ++n) {
      float* dst = static_cast<float*>(dh0->data) + (size_t)n*H;
      const float* src = ws->dXH_cat + (size_t)n*(I+H) + I;
      cudaMemcpyAsync(dst, src, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
  }

  return Status::Ok;
}

} // namespace ai
