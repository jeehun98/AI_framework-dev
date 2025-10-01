// backends/cuda/ops/rnn/launcher.cu
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>   // <-- 추가

#include "backends/cuda/ops/rnn/api.hpp"
// ✅ conv2d와 동일: GEMM 백엔드 직접 호출
#include "backends/cuda/ops/gemm/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif

#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr) do { ::ai::Status _st__=(expr); if(_st__!=::ai::Status::Ok) return _st__; } while(0)
#endif
#ifndef AI_CUDA_TRY
#define AI_CUDA_TRY(expr) do { cudaError_t _e__=(expr); if(_e__!=cudaSuccess) return ::ai::Status::RuntimeError; } while(0)
#endif

namespace ai {

// ===== kernels from rnn/kernels.cu (선언만) =====
Status fill_zero(Tensor& t, StreamHandle s);
Status add_inplace(Tensor& A, const Tensor& B, StreamHandle s);
Status add_out(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s);
Status add_bias_rowwise(Tensor& Y, const Tensor& b, int B, int H, StreamHandle s);
Status tanh_out(const Tensor& X, Tensor& Y, StreamHandle s);
Status tanh_bwd_from_out(const Tensor& Y, const Tensor& dY, Tensor& dZ, StreamHandle s);
Status rowwise_sum_accum(const Tensor& M, Tensor& out, int B, int H, StreamHandle s);
Status transpose_2d(const Tensor& A, Tensor& AT, int M, int N, StreamHandle s);

static inline int64_t numel2(const Tensor& t){
  int64_t n = 1;
  for (auto v : t.desc.shape) n *= v;
  return n;
}

// row-major 2D 뷰 생성
static inline Tensor make_view2d(void* p, int64_t rows, int64_t cols){
  Tensor t;
  t.data         = p;
  t.device       = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype   = DType::F32;
  t.desc.layout  = Layout::RowMajor;
  t.desc.shape   = {rows, cols};
  t.desc.stride  = {cols, 1};
  return t;
}

// [TB,H] 큰 2D에서 행 블록 슬라이스 (row-major)
static inline Tensor row_slice_2d(const Tensor& base, int64_t row0, int64_t rows){
  const int64_t N = base.desc.shape[1];
  auto* p = static_cast<char*>(base.data) + (row0 * N) * sizeof(float);
  return make_view2d(p, rows, N);
}

static inline cudaStream_t to_cuda(StreamHandle s){ return reinterpret_cast<cudaStream_t>(s); }
static inline int div_up(int n, int d){ return (n + d - 1) / d; }

// ===== GEMM 직접 호출 래퍼 =====
// C = opA(A) * opB(B), opA: trans_a ? A^T : A
static inline Status gemm_call(const Tensor& A, const Tensor& B, Tensor& C,
                               bool trans_a, bool trans_b, StreamHandle s){
  GemmAttrs ga{};
  ga.trans_a = trans_a;
  ga.trans_b = trans_b;
  ga.act      = ActKind::None;
  ga.with_bias= false;
  ga.leaky_slope = 0.f;
  return GemmCudaLaunch(A, B, /*Bias*/nullptr, C, ga, s);
}

// 편의 단축
static inline Status gemm_nn(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s){
  return gemm_call(A,B,C,false,false,s);
}
static inline Status gemm_tn(const Tensor& A_T, const Tensor& B, Tensor& C, StreamHandle s){
  return gemm_call(A_T,B,C,true ,false,s);
}
static inline Status gemm_nt(const Tensor& A, const Tensor& B_T, Tensor& C, StreamHandle s){
  return gemm_call(A,B_T,C,false,true ,s);
}

// ===== Forward =====
Status RNNCudaLaunch(const Tensor& X, const Tensor& h0,
                     const Tensor& Wx, const Tensor& Wh,
                     const Tensor* b, Tensor& Hout, Tensor* Zbuf,
                     const RNNAttrs& attrs, StreamHandle s)
{
  const int T = attrs.T, B = attrs.B, I = attrs.I, H = attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;

  const int64_t TB = int64_t(T) * B;

  // workspaces
  void *prez_ptr=nullptr, *tmp_h_ptr=nullptr, *tmp_z_ptr=nullptr;
  const size_t BH = size_t(B)*H*sizeof(float);
  const size_t TBH= size_t(TB)*H*sizeof(float);
  AI_CUDA_TRY(cudaMalloc(&prez_ptr, TBH));     // PreZ_all [TB,H]
  AI_CUDA_TRY(cudaMalloc(&tmp_h_ptr, BH));     // TMP_H    [B,H]
  AI_CUDA_TRY(cudaMalloc(&tmp_z_ptr, BH));     // TMP_Z    [B,H]

  Tensor PreZ_all = make_view2d(prez_ptr, TB, H);
  Tensor TMP_H    = make_view2d(tmp_h_ptr, B,  H);
  Tensor TMP_Z    = make_view2d(tmp_z_ptr, B,  H);

  // 1) PreZ_all = X_all @ Wx
  AI_RETURN_IF_ERROR(gemm_nn(X, Wx, PreZ_all, s));   // [TB,I] x [I,H] -> [TB,H]

  // 2) step loop: Z_t = PreZ_t + hprev@Wh (+b), H_t = tanh(Z_t)
  Tensor hprev = h0;  // [B,H]
  for (int t=0; t<T; ++t){
    int64_t r0 = int64_t(t)*B;
    Tensor PreZ_t = row_slice_2d(PreZ_all, r0, B);   // [B,H]
    Tensor H_t    = row_slice_2d(Hout,     r0, B);   // [B,H]
    Tensor Z_t    = (attrs.save_z && Zbuf) ? row_slice_2d(*Zbuf, r0, B) : TMP_Z;

    // Z_t = PreZ_t
    AI_CUDA_TRY(cudaMemcpyAsync(Z_t.data, PreZ_t.data, BH,
                                cudaMemcpyDeviceToDevice, to_cuda(s)));
    // Z_t += hprev @ Wh
    AI_RETURN_IF_ERROR(gemm_nn(hprev, Wh, TMP_H, s));   // [B,H]x[H,H] -> [B,H]
    AI_RETURN_IF_ERROR(add_inplace(Z_t, TMP_H, s));
    if (b) AI_RETURN_IF_ERROR(add_bias_rowwise(Z_t, *b, B, H, s));

    // H_t = tanh(Z_t)
    AI_RETURN_IF_ERROR(tanh_out(Z_t, H_t, s));
    hprev = H_t;
  }

  cudaFree(tmp_z_ptr);
  cudaFree(tmp_h_ptr);
  cudaFree(prez_ptr);
  return Status::Ok;
}

// ===== Backward =====
Status RNNCudaBackwardLaunch(const Tensor& X, const Tensor& Hout, const Tensor* /*Zbuf*/,
                             const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                             const Tensor& dHout,
                             Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                             const RNNAttrs& attrs, StreamHandle s)
{
  const int T = attrs.T, B = attrs.B, I = attrs.I, H = attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;
  if (!dX || !dh0 || !dWx || !dWh || !dB) return Status::MissingInput;

  const int64_t TB = int64_t(T) * B;
  const size_t BH  = size_t(B)*H*sizeof(float);
  const size_t TBH = size_t(TB)*H*sizeof(float);

  // zero grads
  AI_RETURN_IF_ERROR(fill_zero(*dWx, s));
  AI_RETURN_IF_ERROR(fill_zero(*dWh, s));
  AI_RETURN_IF_ERROR(fill_zero(*dB,  s));

  // --- workspaces ---
  void *dHsum_ptr=nullptr, *dh_next_ptr=nullptr, *dZ_all_ptr=nullptr;
  void *Wx_T_ptr=nullptr, *Wh_T_ptr=nullptr, *Hp_all_ptr=nullptr;
  AI_CUDA_TRY(cudaMalloc(&dHsum_ptr,  BH));           // [B,H]
  AI_CUDA_TRY(cudaMalloc(&dh_next_ptr, BH));          // [B,H]
  AI_CUDA_TRY(cudaMalloc(&dZ_all_ptr,  TBH));         // [TB,H]
  AI_CUDA_TRY(cudaMalloc(&Wx_T_ptr,    size_t(H)*I*sizeof(float)));  // [H,I]
  AI_CUDA_TRY(cudaMalloc(&Wh_T_ptr,    size_t(H)*H*sizeof(float)));  // [H,H]
  AI_CUDA_TRY(cudaMalloc(&Hp_all_ptr,  TBH));         // [TB,H] (Hprev_all)

  Tensor dHsum    = make_view2d(dHsum_ptr,  B,  H);
  Tensor dh_next  = make_view2d(dh_next_ptr, B, H);
  Tensor dZ_all   = make_view2d(dZ_all_ptr, TB, H);
  Tensor Wx_T     = make_view2d(Wx_T_ptr,   H,  I);
  Tensor Wh_T     = make_view2d(Wh_T_ptr,   H,  H);
  Tensor Hprev_all= make_view2d(Hp_all_ptr, TB, H);

  AI_RETURN_IF_ERROR(fill_zero(dh_next, s));

  // 사전 전치
  AI_RETURN_IF_ERROR(transpose_2d(Wx, Wx_T, I, H, s));   // [I,H]->[H,I]
  AI_RETURN_IF_ERROR(transpose_2d(Wh, Wh_T, H, H, s));   // [H,H]->[H,H]

  // 1) 루프: dZ_all 채우고, dh_next만 recurrence로 업데이트
  for (int t=T-1; t>=0; --t){
    int64_t r0 = int64_t(t)*B;
    Tensor H_t   = row_slice_2d(Hout,  r0, B);   // [B,H]
    Tensor dH_t  = row_slice_2d(dHout, r0, B);   // [B,H]
    Tensor dZ_t  = row_slice_2d(dZ_all,r0, B);   // [B,H]

    // dHsum = dH_t + dh_next
    AI_RETURN_IF_ERROR(add_out(dH_t, dh_next, dHsum, s));
    // dZ_t = dHsum * (1 - H_t^2)
    AI_RETURN_IF_ERROR(tanh_bwd_from_out(H_t, dHsum, dZ_t, s));

    // dh_next = dZ_t @ Wh^T
    AI_RETURN_IF_ERROR(gemm_nn(dZ_t, Wh_T, dh_next, s));
  }

  // 2) dB = sum_rows(dZ_all)  (TB, H)
  AI_RETURN_IF_ERROR(rowwise_sum_accum(dZ_all, *dB, (int)TB, H, s));

  // 3) Hprev_all 구성: 첫 블록은 h0, 나머지는 Hout[t-1]
  {
    // Hprev_all[0:B,:] = h0
    AI_CUDA_TRY(cudaMemcpyAsync(Hprev_all.data,
                                h0.data, BH,
                                cudaMemcpyDeviceToDevice, to_cuda(s)));
    // for t=1..T-1: copy Hout[(t-1)*B:(t)*B] -> Hprev_all[t*B:(t+1)*B]
    for (int t=1; t<T; ++t){
      int64_t dst_r0 = int64_t(t)*B;
      int64_t src_r0 = int64_t(t-1)*B;
      Tensor dst = row_slice_2d(Hprev_all, dst_r0, B);
      Tensor src = row_slice_2d(Hout,      src_r0, B);
      AI_CUDA_TRY(cudaMemcpyAsync(dst.data, src.data, BH,
                                  cudaMemcpyDeviceToDevice, to_cuda(s)));
    }
  }

  // 4) 큰 GEMM 3개
  // (a) dWx = X_all^T @ dZ_all  -> [I,H]
  {
    void *Xt_T_ptr=nullptr; AI_CUDA_TRY(cudaMalloc(&Xt_T_ptr, size_t(I)*TB*sizeof(float)));
    Tensor Xt_T = make_view2d(Xt_T_ptr, I, TB);
    AI_RETURN_IF_ERROR(transpose_2d(X, Xt_T, (int)TB, I, s));
    AI_RETURN_IF_ERROR(gemm_nn(Xt_T, dZ_all, *dWx, s));  // [I,TB]x[TB,H] -> [I,H]
    cudaFree(Xt_T_ptr);
  }

  // (b) dX_all = dZ_all @ Wx^T   -> [TB,I]
  {
    AI_RETURN_IF_ERROR(gemm_nn(dZ_all, Wx_T, *dX, s));   // [TB,H]x[H,I] -> [TB,I]
  }

  // (c) dWh = Hprev_all^T @ dZ_all -> [H,H]
  {
    void *HpT_ptr=nullptr; AI_CUDA_TRY(cudaMalloc(&HpT_ptr, size_t(H)*TB*sizeof(float)));
    Tensor Hp_T = make_view2d(HpT_ptr, H, TB);
    AI_RETURN_IF_ERROR(transpose_2d(Hprev_all, Hp_T, (int)TB, H, s));
    AI_RETURN_IF_ERROR(gemm_nn(Hp_T, dZ_all, *dWh, s));  // [H,TB]x[TB,H] -> [H,H]
    cudaFree(HpT_ptr);
  }

  // 5) dh0 = dh_next
  AI_CUDA_TRY(cudaMemcpyAsync(dh0->data, dh_next.data, BH,
                              cudaMemcpyDeviceToDevice, to_cuda(s)));

  cudaFree(Hp_all_ptr);
  cudaFree(Wh_T_ptr);
  cudaFree(Wx_T_ptr);
  cudaFree(dZ_all_ptr);
  cudaFree(dh_next_ptr);
  cudaFree(dHsum_ptr);
  return Status::Ok;
}



} // namespace ai
