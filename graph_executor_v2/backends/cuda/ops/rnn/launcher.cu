// backends/cuda/ops/rnn/launcher.cu
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>

#include "backends/cuda/ops/rnn/api.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
  #include "ai/dispatch.hpp"
#endif

#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr) do { ::ai::Status _st__=(expr); if(_st__!=::ai::Status::Ok) return _st__; } while(0)
#endif
#ifndef AI_CUDA_TRY
#define AI_CUDA_TRY(expr) do { cudaError_t _e__=(expr); if(_e__!=cudaSuccess) return ::ai::Status::RuntimeError; } while(0)
#endif

namespace ai {

// ----------------------- small utils -----------------------
static inline Tensor make_view2d(void* p, int64_t rows, int64_t cols){
  Tensor t; t.data=p; t.device=Device::CUDA; t.device_index=0;
  t.desc={DType::F32, Layout::RowMajor, {rows, cols}, {cols, 1}};
  return t;
}
static inline Tensor row_slice_2d(const Tensor& base, int64_t row0, int64_t rows){
  const int64_t N = base.desc.shape[1];
  auto* p = static_cast<char*>(base.data) + (row0 * N) * sizeof(float);
  return make_view2d(p, rows, N);
}
static inline bool is2d_f32_cuda_rowmajor(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==2;
}
static inline cudaStream_t to_cuda(StreamHandle s){ return reinterpret_cast<cudaStream_t>(s); }

// ----------------------- GEMM wrappers ---------------------
// NOTE: GemmCudaLaunch는 transpose 미지원. FWD에서만 NN 경로로 사용.
static inline Status gemm_call(const Tensor& A, const Tensor& B, Tensor& C, bool ta, bool tb, StreamHandle s){
  GemmAttrs ga{}; ga.trans_a=ta; ga.trans_b=tb; ga.act=ActKind::None; ga.with_bias=false; ga.leaky_slope=0.f;
  return GemmCudaLaunch(A, B, /*Bias*/nullptr, C, ga, s);
}
static inline Status gemm_nn(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s){ return gemm_call(A,B,C,false,false,s); }
// tn/nt는 더 이상 사용하지 않음 (transpose 금지)

// ===========================================================
//                           Forward
// ===========================================================
Status RNNCudaLaunch(const Tensor& X, const Tensor& h0,
                     const Tensor& Wx, const Tensor& Wh,
                     const Tensor* b, Tensor& Hout, Tensor* Zbuf,
                     const RNNAttrs& attrs, StreamHandle s,
                     const RNNWorkspaceFwd* ws_fwd /*=nullptr*/)
{
  const int T=attrs.T, B=attrs.B, I=attrs.I, H=attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;

  // dtype/layout/device/rank checks
  if (!is2d_f32_cuda_rowmajor(X)    ||
      !is2d_f32_cuda_rowmajor(Hout) ||
      !is2d_f32_cuda_rowmajor(Wx)   ||
      !is2d_f32_cuda_rowmajor(Wh)) return Status::Invalid;

  // shape checks
  if (X.desc.shape[0] != (int64_t)T*B || X.desc.shape[1]!=I) return Status::ShapeMismatch;
  if (Hout.desc.shape[0]!=(int64_t)T*B || Hout.desc.shape[1]!=H) return Status::ShapeMismatch;
  if (Wx.desc.shape[0]!=I || Wx.desc.shape[1]!=H) return Status::ShapeMismatch;
  if (Wh.desc.shape[0]!=H || Wh.desc.shape[1]!=H) return Status::ShapeMismatch;

  if (!(h0.device==Device::CUDA && h0.desc.dtype==DType::F32 &&
        h0.desc.layout==Layout::RowMajor && h0.desc.shape.size()==2 &&
        h0.desc.shape[0]==B && h0.desc.shape[1]==H)) return Status::ShapeMismatch;

  if (b){
    if (!(b->device==Device::CUDA && b->desc.dtype==DType::F32 &&
          b->desc.layout==Layout::RowMajor && b->desc.shape.size()==1 &&
          b->desc.shape[0]==H)) return Status::ShapeMismatch;
  }

  // save_z=true → Zbuf 필수
  if (attrs.save_z){
    if (!Zbuf) return Status::Invalid;
    if (!is2d_f32_cuda_rowmajor(*Zbuf) ||
        Zbuf->desc.shape[0]!=(int64_t)T*B || Zbuf->desc.shape[1]!=H)
      return Status::ShapeMismatch;
  }

  // 캡처-세이프: 내부 할당 금지, ws_fwd 필수
  if (!ws_fwd || !ws_fwd->PreZ_all || !ws_fwd->TMP_H || !ws_fwd->TMP_Z){
#ifdef AI_DEBUG
    fprintf(stderr, "[RNN][FWD] workspace required (PreZ_all/TMP_H/TMP_Z)\n");
#endif
    return Status::Invalid;
  }

  const int64_t TB = int64_t(T)*B;
  const size_t  BH = size_t(B)*H*sizeof(float);

  Tensor PreZ_all = make_view2d(ws_fwd->PreZ_all, TB, H);
  Tensor TMP_H    = make_view2d(ws_fwd->TMP_H,    B,  H);
  Tensor TMP_Z    = make_view2d(ws_fwd->TMP_Z,    B,  H);

  // 1) PreZ_all = X @ Wx
  AI_RETURN_IF_ERROR(gemm_nn(X, Wx, PreZ_all, s));

  // 2) time-step loop
  Tensor hprev = h0;
  for (int t=0; t<T; ++t){
    const int64_t r0 = int64_t(t)*B;
    Tensor PreZ_t = row_slice_2d(PreZ_all, r0, B);
    Tensor H_t    = row_slice_2d(Hout,     r0, B);
    Tensor Z_t    = (attrs.save_z ? row_slice_2d(*Zbuf, r0, B) : TMP_Z);

    // Z_t = PreZ_t + hprev@Wh + b
    AI_CUDA_TRY(cudaMemcpyAsync(Z_t.data, PreZ_t.data, BH,
                                cudaMemcpyDeviceToDevice, to_cuda(s)));
    AI_RETURN_IF_ERROR(gemm_nn(hprev, Wh, TMP_H, s));     // hprev @ Wh
    AI_RETURN_IF_ERROR(add_inplace(Z_t, TMP_H, s));       // + TMP_H
    if (b) AI_RETURN_IF_ERROR(add_bias_rowwise(Z_t, *b, B, H, s));

    // H_t = tanh(Z_t)
    AI_RETURN_IF_ERROR(tanh_out(Z_t, H_t, s));

    hprev = H_t;
  }

  return Status::Ok;
}

// ===========================================================
//                           Backward
// ===========================================================
Status RNNCudaBackwardLaunch(const Tensor& X, const Tensor& Hout, const Tensor* Zbuf,
                             const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                             const Tensor& dHout,
                             Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                             const RNNAttrs& attrs, StreamHandle s,
                             const RNNWorkspaceBwd* ws_bwd /*=nullptr*/)
{
  const int T=attrs.T, B=attrs.B, I=attrs.I, H=attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;

  // base checks
  if (!is2d_f32_cuda_rowmajor(X)     ||
      !is2d_f32_cuda_rowmajor(Hout)  ||
      !is2d_f32_cuda_rowmajor(Wx)    ||
      !is2d_f32_cuda_rowmajor(Wh)    ||
      !is2d_f32_cuda_rowmajor(dHout)) return Status::Invalid;

  if (X.desc.shape[0]!=(int64_t)T*B || X.desc.shape[1]!=I) return Status::ShapeMismatch;
  if (Hout.desc.shape[0]!=(int64_t)T*B || Hout.desc.shape[1]!=H) return Status::ShapeMismatch;
  if (dHout.desc.shape != Hout.desc.shape) return Status::ShapeMismatch;
  if (Wx.desc.shape[0]!=I || Wx.desc.shape[1]!=H) return Status::ShapeMismatch;
  if (Wh.desc.shape[0]!=H || Wh.desc.shape[1]!=H) return Status::ShapeMismatch;

  // optional grads shape checks (nullable 허용)
  if (dX  && (!is2d_f32_cuda_rowmajor(*dX)  || dX->desc.shape[0]!=(int64_t)T*B || dX->desc.shape[1]!=I)) return Status::ShapeMismatch;
  if (dWx && (!is2d_f32_cuda_rowmajor(*dWx) || dWx->desc.shape[0]!=I || dWx->desc.shape[1]!=H))          return Status::ShapeMismatch;
  if (dWh && (!is2d_f32_cuda_rowmajor(*dWh) || dWh->desc.shape[0]!=H || dWh->desc.shape[1]!=H))          return Status::ShapeMismatch;
  if (dB){
    if (!(dB->device==Device::CUDA && dB->desc.dtype==DType::F32 &&
          dB->desc.layout==Layout::RowMajor && dB->desc.shape.size()==1 &&
          dB->desc.shape[0]==H)) return Status::ShapeMismatch;
  }

  if (!(h0.device==Device::CUDA && h0.desc.dtype==DType::F32 &&
        h0.desc.layout==Layout::RowMajor && h0.desc.shape.size()==2 &&
        h0.desc.shape[0]==B && h0.desc.shape[1]==H)) return Status::ShapeMismatch;

  // save_z=true → Zbuf 필수(경로 고정, 여기선 미사용이더라도 일관성 위해 검사)
  if (attrs.save_z){
    if (!Zbuf || !is2d_f32_cuda_rowmajor(*Zbuf) ||
        Zbuf->desc.shape[0]!=(int64_t)T*B || Zbuf->desc.shape[1]!=H) return Status::Invalid;
  }

  // 캡처-세이프: ws_bwd 필수
  if (!ws_bwd || !ws_bwd->dHsum || !ws_bwd->dh_next || !ws_bwd->dZ_all || !ws_bwd->Hprev_all){
#ifdef AI_DEBUG
    fprintf(stderr, "[RNN][BWD] workspace required (dHsum/dh_next/dZ_all/Hprev_all)\n");
#endif
    return Status::Invalid;
  }

  const int64_t TB = int64_t(T)*B;
  const size_t  BH = size_t(B)*H*sizeof(float);

  // zero/init grads (요청된 항목만)
  if (dWx) AI_RETURN_IF_ERROR(fill_zero(*dWx, s));
  if (dWh) AI_RETURN_IF_ERROR(fill_zero(*dWh, s));
  if (dB ) AI_RETURN_IF_ERROR(fill_zero(*dB , s));
  // dX, dh0는 아래서 계산 시 채움

  Tensor dHsum     = make_view2d(ws_bwd->dHsum,     B,  H);
  Tensor dh_next   = make_view2d(ws_bwd->dh_next,   B,  H);
  Tensor dZ_all    = make_view2d(ws_bwd->dZ_all,    TB, H);
  Tensor Hprev_all = make_view2d(ws_bwd->Hprev_all, TB, H);

  AI_RETURN_IF_ERROR(fill_zero(dh_next, s));

  // 1) dZ_all 채우기 & dh_next 전파 (tanh'는 H로 계산)
  for (int t=T-1; t>=0; --t){
    const int64_t r0 = int64_t(t)*B;
    Tensor H_t  = row_slice_2d(Hout,  r0, B);
    Tensor dH_t = row_slice_2d(dHout, r0, B);
    Tensor dZ_t = row_slice_2d(dZ_all,r0, B);

    AI_RETURN_IF_ERROR(add_out(dH_t, dh_next, dHsum, s));        // dHsum = dH_t + dh_next
    AI_RETURN_IF_ERROR(tanh_bwd_from_out(H_t, dHsum, dZ_t, s));  // dZ_t = (1 - H_t^2) * dHsum

    // dh_next = dZ_t @ Wh^T  (transpose 금지 → GemmCudaBackward로 대체)
    {
      GemmAttrs ga_none{}; ga_none.act=ActKind::None; ga_none.with_bias=false;
      ga_none.trans_a=false; ga_none.trans_b=false; ga_none.leaky_slope=0.f;

      // dummy A: 값은 사용되지 않음(모양만 [B,H]) — 이미 존재하는 버퍼를 뷰로 재사용
      Tensor dummyA = row_slice_2d(Hout, r0, B); // [B,H]

      AI_RETURN_IF_ERROR(GemmCudaBackward(
        /*A*/ dummyA, /*B*/ Wh, /*C*/ nullptr,
        /*gY*/ dZ_t,  /*Z*/ dZ_t,
        /*gA*/ &dh_next, /*gB*/ nullptr, /*gC*/ nullptr, /*gBias*/ nullptr,
        ga_none, s, /*ws*/ nullptr));
    }
  }

  // 2) Hprev_all(0)=h0, Hprev_all(t)=Hout(t-1)
  AI_CUDA_TRY(cudaMemcpyAsync(Hprev_all.data, h0.data, BH,
                              cudaMemcpyDeviceToDevice, to_cuda(s)));
  for (int t=1; t<T; ++t){
    const int64_t dst_r0 = int64_t(t)*B, src_r0 = int64_t(t-1)*B;
    Tensor dst = row_slice_2d(Hprev_all, dst_r0, B);
    Tensor src = row_slice_2d(Hout,      src_r0, B);
    AI_CUDA_TRY(cudaMemcpyAsync(dst.data, src.data, BH,
                                cudaMemcpyDeviceToDevice, to_cuda(s)));
  }

  // 3) 파라미터/입력 그라디언트 (transpose 금지 → GemmCudaBackward로 처리)
  {
    GemmAttrs ga_none{}; ga_none.act=ActKind::None; ga_none.with_bias=false;
    ga_none.trans_a=false; ga_none.trans_b=false; ga_none.leaky_slope=0.f;

    // (a) dX, dWx, dB  — dX = dZ_all @ Wx^T, dWx = X^T @ dZ_all, dBias(PerN) = sum_rows(dZ_all)
    AI_RETURN_IF_ERROR(GemmCudaBackward(
      /*A*/ X, /*B*/ Wx, /*C*/ nullptr,
      /*gY*/ dZ_all, /*Z*/ dZ_all,
      /*gA*/ dX,     /*gB*/ dWx, /*gC*/ nullptr, /*gBias*/ dB,
      ga_none, s, /*ws*/ nullptr));

    // (b) dWh = Hprev_all^T @ dZ_all
    if (dWh){
      AI_RETURN_IF_ERROR(GemmCudaBackward(
        /*A*/ Hprev_all, /*B*/ Wh, /*C*/ nullptr,
        /*gY*/ dZ_all,   /*Z*/ dZ_all,
        /*gA*/ nullptr,  /*gB*/ dWh, /*gC*/ nullptr, /*gBias*/ nullptr,
        ga_none, s, /*ws*/ nullptr));
    }
  }

  // 4) dh0
  if (dh0){
    AI_CUDA_TRY(cudaMemcpyAsync(dh0->data, dh_next.data, BH,
                                cudaMemcpyDeviceToDevice, to_cuda(s)));
  }

  return Status::Ok;
}

} // namespace ai
