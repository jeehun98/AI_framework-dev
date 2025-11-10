// backends/cuda/ops/rnn/launcher.cu
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cassert>

#include "backends/cuda/ops/gemm/api.hpp"   // GemmCudaLaunch / GemmCudaBackward
#include "backends/cuda/ops/rnn/api.hpp"
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

// ---------- 커널 전방 선언 ----------
void apply_act_rows_launcher(const float* Z, float* Y,
                             int M, int N, int act_code, float slope, cudaStream_t s);
void apply_dact_rows_launcher(const float* gy_post, const float* Z, float* gy,
                              int M, int N, int act_code, float slope, cudaStream_t s);
void add_dhnext_into_grows_launcher(const float* dXH, float* dG,
                                    int N, int I, int H, cudaStream_t s);
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s);
void add_transpose_into_launcher(float* dWcat, const float* Tmp_H_IpH,
                                 int IpH, int H, cudaStream_t s);
void reduce_db_rows_NH_launcher(const float* G, float* db, int N, int H, cudaStream_t s);

// ---------- 검증 헬퍼 (새 ai_validate 기반) ----------
static inline Status expect_rowmajor_3d_f32(const Tensor& T) {
  return expect_rowmajor_nd(T, /*rank=*/3, {/*allowed dtypes*/ DType::F32});
}
static inline Status expect_rowmajor_2d_f32(const Tensor& T) {
  return expect_rowmajor_nd(T, /*rank=*/2, {DType::F32});
}
static inline Status expect_rowmajor_2d_f32_shape(const Tensor& T, int64_t D0, int64_t D1) {
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(T));
  if (T.desc.shape[0] != D0 || T.desc.shape[1] != D1) return Status::ShapeMismatch;
  return Status::Ok;
}
static inline Status expect_rowmajor_1d_f32_len(const Tensor& T, int64_t L) {
  AI_RETURN_IF_ERROR(expect_rowmajor_nd(T, /*rank=*/1, {DType::F32}));
  return (T.desc.shape[0] == L) ? Status::Ok : Status::ShapeMismatch;
}

// ---------- 간단 검증 래퍼 ----------
static inline Status validate_fwd(const Tensor& X, const Tensor& Wx, const Tensor& Wh,
                                  const Tensor* B, const Tensor& h0, Tensor& Y,
                                  const RnnAttrs& a, bool want_z, const Tensor* Z_saved) {
  // X:[N,T,I], Wx:[I,H], Wh:[H,H], h0:[N,H], Y:[N,T,H] (모두 f32, row-major, 연속)
  AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(X));
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(Wx));
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(Wh));
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(h0));
  AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(Y));

  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];
  const int64_t H = Wx.desc.shape[1];

  // Wx:[I,H], Wh:[H,H]
  if (Wx.desc.shape[0] != I) return Status::ShapeMismatch;
  if (Wh.desc.shape[0] != H || Wh.desc.shape[1] != H) return Status::ShapeMismatch;

  // Y:[N,T,H]
  if (Y.desc.shape[0] != N || Y.desc.shape[1] != T || Y.desc.shape[2] != H) return Status::ShapeMismatch;

  // h0:[N,H]
  if (h0.desc.shape[0] != N || h0.desc.shape[1] != H) return Status::ShapeMismatch;

  // bias
  if (a.with_bias) {
    if (!B) return Status::MissingInput;
    AI_RETURN_IF_ERROR(expect_rowmajor_1d_f32_len(*B, H));
  }

  // save Z
  if (want_z) {
    if (!Z_saved) return Status::MissingInput;
    AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(*Z_saved));
    if (Z_saved->desc.shape[0] != N || Z_saved->desc.shape[1] != T || Z_saved->desc.shape[2] != H)
      return Status::ShapeMismatch;
    if (Z_saved->data == Y.data) return Status::Invalid; // alias 금지
  }

  return Status::Ok;
}

static inline Status validate_bwd(const Tensor& X, const Tensor& Wx, const Tensor& Wh,
                                  const Tensor& h0, const Tensor& dY_post, const Tensor& Z,
                                  const Tensor* dWx, const Tensor* dWh, const Tensor* dB,
                                  const Tensor* dh0, const Tensor* dX, const RnnAttrs& a) {
  AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(X));       // [N,T,I]
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(Wx));      // [I,H]
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(Wh));      // [H,H]
  AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32(h0));      // [N,H]
  AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(dY_post)); // [N,T,H]
  AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(Z));       // [N,T,H]

  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];
  const int64_t H = Wx.desc.shape[1];

  // Z:[N,T,H]
  if (Z.desc.shape[0] != N || Z.desc.shape[1] != T || Z.desc.shape[2] != H) return Status::ShapeMismatch;

  // dWx:[I,H], dWh:[H,H], dB:[H], dh0:[N,H], dX:[N,T,I]
  if (dWx) AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32_shape(*dWx, I, H));
  if (dWh) AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32_shape(*dWh, H, H));
  if (a.with_bias && dB) AI_RETURN_IF_ERROR(expect_rowmajor_1d_f32_len(*dB, H));
  if (dh0) AI_RETURN_IF_ERROR(expect_rowmajor_2d_f32_shape(*dh0, N, H));
  if (dX) {
    AI_RETURN_IF_ERROR(expect_rowmajor_3d_f32(*dX));
    if (dX->desc.shape[0] != N || dX->desc.shape[1] != T || dX->desc.shape[2] != I) return Status::ShapeMismatch;
  }
  return Status::Ok;
}

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
  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];
  const int64_t H = Wx.desc.shape[1];
  const bool want_z = a.save_z && (Z_saved != nullptr);

  AI_RETURN_IF_ERROR(validate_fwd(X, Wx, Wh, B, h0, Y, a, want_z, Z_saved));
  if (!ws || !ws->XH_cat || !ws->Y_rows || !ws->W_cat || (want_z && !ws->Z_rows))
    return Status::MissingInput;

  auto s = as_cuda_stream(stream);

  // W_cat = [Wx ; Wh] : [I+H, H]
  AI_RETURN_IF_ERROR(memcpy_d2d_async(ws->W_cat, Wx.data, (size_t)I * H * sizeof(float), s));
  AI_RETURN_IF_ERROR(memcpy_d2d_async(ws->W_cat + (size_t)I * H, Wh.data, (size_t)H * H * sizeof(float), s));

  ai::GemmAttrs g{};
  g.act          = a.act;
  g.leaky_slope  = a.leaky_slope;
  g.with_bias    = (B && a.with_bias);
  g.save_z       = want_z;

  const ai::Tensor* BiasPtr = nullptr;
  ai::Tensor tBias{};
  if (g.with_bias) {
    tBias.data        = const_cast<void*>(B->data);
    tBias.device      = ai::Device::CUDA; tBias.device_index = 0;
    tBias.desc.dtype  = ai::DType::F32;   tBias.desc.layout  = ai::Layout::RowMajor;
    tBias.desc.shape  = { H };            tBias.desc.stride  = { 1 };
    BiasPtr = &tBias;
  }

  Tensor tA{}, tB{}, tO{}, tZcap{};
  const float* h_prev = static_cast<const float*>(h0.data);

  const size_t X_pitch_elems_dst = (size_t)(I + H);
  const size_t X_pitch_elems_src = (size_t)(T * I);
  const size_t H_pitch_elems     = (size_t)H;

  for (int64_t t = 0; t < T; ++t) {
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
    tA = Tensor{ ws->XH_cat, {DType::F32, Layout::RowMajor, {N, I + H}, {I + H, 1}}, Device::CUDA, 0 };
    tB = Tensor{ ws->W_cat,  {DType::F32, Layout::RowMajor, {I + H, H}, {H, 1}},     Device::CUDA, 0 };
    tO = Tensor{ ws->Y_rows, {DType::F32, Layout::RowMajor, {N, H},     {H, 1}},     Device::CUDA, 0 };
    if (want_z) tZcap = Tensor{ ws->Z_rows, {DType::F32, Layout::RowMajor, {N, H}, {H, 1}}, Device::CUDA, 0 };

    AI_RETURN_IF_ERROR(GemmCudaLaunch(tA, tB, BiasPtr, tO, g, stream, (want_z ? &tZcap : nullptr)));

    // Y_rows → Y[:,t,:]
    {
      void* dst = static_cast<float*>(Y.data) + (size_t)t * H;
      const void* src = ws->Y_rows;
      cudaMemcpy2DAsync(dst, (size_t)(T * H) * sizeof(float),
                        src, (size_t)H * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }
    // Z_rows → Z[:,t,:]
    if (want_z) {
      void* dst = static_cast<float*>(Z_saved->data) + (size_t)t * H;
      const void* src = ws->Z_rows;
      cudaMemcpy2DAsync(dst, (size_t)(T * H) * sizeof(float),
                        src, (size_t)H * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    h_prev = ws->Y_rows; // 다음 타임스텝 입력
  }

  AI_CUDA_CHECK_LAUNCH();
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
  AI_RETURN_IF_ERROR(validate_bwd(X, Wx, Wh, h0, dY_post, Z, dWx, dWh, dB, dh0, dX, a));
  if (!ws || !ws->XH_cat || !ws->G_rows || !ws->Z_rows || !ws->W_cat ||
      !ws->dXH_cat || !ws->dWcat || !ws->TmpW)
    return Status::MissingInput;

  const int64_t N = X.desc.shape[0];
  const int64_t T = X.desc.shape[1];
  const int64_t I = X.desc.shape[2];
  const int64_t H = Wx.desc.shape[1];

  auto s = as_cuda_stream(stream);

  // zero grads
  if (dWx) AI_RETURN_IF_ERROR(ai_memset_async(dWx->data, 0, sizeof(float) * (size_t)I * H, s));
  if (dWh) AI_RETURN_IF_ERROR(ai_memset_async(dWh->data, 0, sizeof(float) * (size_t)H * H, s));
  if (dB && a.with_bias) AI_RETURN_IF_ERROR(ai_memset_async(dB->data, 0, sizeof(float) * (size_t)H, s));
  if (dh0) AI_RETURN_IF_ERROR(ai_memset_async(dh0->data, 0, sizeof(float) * (size_t)N * H, s));
  if (dX)  AI_RETURN_IF_ERROR(ai_memset_async(dX->data,  0, sizeof(float) * (size_t)N * T * I, s));
  AI_RETURN_IF_ERROR(ai_memset_async(ws->dWcat,   0, sizeof(float) * (size_t)(I + H) * H, s));
  AI_RETURN_IF_ERROR(ai_memset_async(ws->dXH_cat, 0, sizeof(float) * (size_t)N * (I + H), s));

  // W_cat = [Wx ; Wh]
  AI_RETURN_IF_ERROR(memcpy_d2d_async(ws->W_cat, Wx.data, (size_t)I * H * sizeof(float), s));
  AI_RETURN_IF_ERROR(memcpy_d2d_async(ws->W_cat + (size_t)I * H, Wh.data, (size_t)H * H * sizeof(float), s));

  ai::GemmAttrs g{};
  g.act         = a.act;
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = (a.with_bias && dB);

  const size_t X_pitch_elems_dst = (size_t)(I + H);
  const size_t X_pitch_elems_src = (size_t)(T * I);
  const size_t H_pitch_elems     = (size_t)H;
  const size_t TH_pitch_elems    = (size_t)(T * H);

  for (int64_t t = T - 1; t >= 0; --t) {
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
      {
        const void* src_z = static_cast<const float*>(Z.data) + (size_t)(t - 1) * H;
        void* dst_z = ws->Z_rows;
        cudaMemcpy2DAsync(dst_z, H_pitch_elems * sizeof(float),
                          src_z, TH_pitch_elems * sizeof(float),
                          H * sizeof(float), (size_t)N,
                          cudaMemcpyDeviceToDevice, s);
      }
      apply_act_rows_launcher(ws->Z_rows, ws->Z_rows, (int)N, (int)H, (int)a.act, a.leaky_slope, s);
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
    // + dh_next 누적
    add_dhnext_into_grows_launcher(ws->dXH_cat, ws->G_rows, (int)N, (int)I, (int)H, s);

    // Z_rows <- Z[:,t,:]
    {
      const void* src_z = static_cast<const float*>(Z.data) + (size_t)t * H;
      void* dst_z = ws->Z_rows;
      cudaMemcpy2DAsync(dst_z, H_pitch_elems * sizeof(float),
                        src_z, TH_pitch_elems * sizeof(float),
                        H * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    // 텐서 뷰
    Tensor tA   { ws->XH_cat, {DType::F32, Layout::RowMajor, {N, I + H}, {I + H, 1}}, Device::CUDA, 0 };
    Tensor tB   { ws->W_cat,  {DType::F32, Layout::RowMajor, {I + H, H}, {H, 1}},     Device::CUDA, 0 };
    Tensor tGY  { ws->G_rows, {DType::F32, Layout::RowMajor, {N, H},     {H, 1}},     Device::CUDA, 0 };
    Tensor tZ   { ws->Z_rows, {DType::F32, Layout::RowMajor, {N, H},     {H, 1}},     Device::CUDA, 0 };
    Tensor tGA  { ws->dXH_cat,{DType::F32, Layout::RowMajor, {N, I + H}, {I + H, 1}}, Device::CUDA, 0 };
    Tensor tGB  { ws->TmpW,   {DType::F32, Layout::RowMajor, {I + H, H}, {H, 1}},     Device::CUDA, 0 };
    Tensor* tGC = nullptr;
    Tensor* tGBias = (dB && a.with_bias) ? dB : nullptr;

    AI_RETURN_IF_ERROR(GemmCudaBackward(
      tA, tB, /*C*/nullptr, tGY, tZ,
      /*gA*/ &tGA, /*gB*/ &tGB, /*gC*/ tGC, /*gBias*/ tGBias,
      g, stream, /*ws=*/nullptr));

    // dX[:,t,:] = dXH_cat[:, :I]
    if (dX) {
      void* dst = static_cast<float*>(dX->data) + (size_t)t * I;
      const void* src = ws->dXH_cat;
      cudaMemcpy2DAsync(dst, (size_t)(T * I) * sizeof(float),
                        src, (size_t)(I + H) * sizeof(float),
                        I * sizeof(float), (size_t)N,
                        cudaMemcpyDeviceToDevice, s);
    }

    // dWcat += gB  (gB: [I+H,H])  →  transpose(H,I+H) 후 누적
    transpose_kernel_launcher(ws->TmpW, ws->Z_rows, /*M=*/(int)(I + H), /*N=*/(int)H, s); // Z_rows := gB^T [H, I+H]
    add_transpose_into_launcher(ws->dWcat, ws->Z_rows, (int)(I + H), (int)H, s);
  } // for t

  // split dWcat -> dWx, dWh
  if (dWx) AI_RETURN_IF_ERROR(memcpy_d2d_async(dWx->data, ws->dWcat, (size_t)I * H * sizeof(float), s));
  if (dWh) AI_RETURN_IF_ERROR(memcpy_d2d_async(dWh->data, ws->dWcat + (size_t)I * H, (size_t)H * H * sizeof(float), s));

  // dh0 = dXH_cat[:, I:]
  if (dh0) {
    void* dst = dh0->data;
    const void* src = ws->dXH_cat + I;
    cudaMemcpy2DAsync(dst, (size_t)H * sizeof(float),
                      src, (size_t)(I + H) * sizeof(float),
                      H * sizeof(float), (size_t)N,
                      cudaMemcpyDeviceToDevice, s);
  }

  AI_CUDA_CHECK_LAUNCH();
  return Status::Ok;
}

} // namespace ai
