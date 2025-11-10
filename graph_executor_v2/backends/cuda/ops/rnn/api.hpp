#pragma once
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

// ========== 속성 ==========
struct RnnAttrs {
  ai::ActKind act{ai::ActKind::None};
  float       leaky_slope{0.01f};
  bool        with_bias{false};
  bool        save_z{true};
};

// ========== 캡처-세이프 워크스페이스 (Forward) ==========
struct RnnWorkspaceFwd {
  // shapes assume: X:[N,T,I], Wx:[I,H], Wh:[H,H], Y:[N,T,H]
  // [필수]
  float* XH_cat {nullptr}; // [N, I+H] : concat([X_t, h_{t-1}])
  float* Y_rows {nullptr}; // [N, H]   : GEMM output per time
  float* W_cat  {nullptr}; // [I+H, H] : packed [Wx; Wh] (런타임 pack)
  // [옵션] save_z==true
  float* Z_rows {nullptr}; // [N, H]   : pre-activation rows
};

// ========== 캡처-세이프 워크스페이스 (Backward) ==========
struct RnnWorkspaceBwd {
  // [필수]
  float* XH_cat {nullptr}; // [N, I+H]
  float* G_rows {nullptr}; // [N, H]
  float* Z_rows {nullptr}; // [N, H]
  float* W_cat  {nullptr}; // [I+H, H]
  float* dXH_cat{nullptr}; // [N, I+H]
  float* dWcat  {nullptr}; // [I+H, H]
  float* TmpW   {nullptr}; // [I+H, H]
};

// ========== 커널 런처 (kernels.cu 에서 제공) ==========
// (아래 심볼명은 통일된 공용 시그니처)
void apply_dact_rows_launcher(const float* gy_post, const float* Z_rows, float* gy_rows,
                              int M, int N, int act_code, float slope, cudaStream_t s);
void add_rows_strided_launcher(float* A_MN, const float* B_Mstride,
                               int M, int N, int strideB, int offsetB, cudaStream_t s);
void reduce_db_rows_kernel_launcher(const float* G_MN, float* db_N,
                                    int M, int N, cudaStream_t s);
void kadd_vec_launcher(float* A, const float* B, int n, cudaStream_t s);
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s);
void pack_wcat_from_wx_wh_launcher(const float* Wx, const float* Wh, float* Wcat,
                                   int I, int H, cudaStream_t s);
void apply_act_rows_launcher(const float* Z_rows, float* H_rows,
                             int M, int N, int act_code, float slope, cudaStream_t s);

// ========== Forward ==========
Status RnnCudaLaunch(const Tensor& X,   // [N,T,I]
                     const Tensor& Wx,  // [I,H]
                     const Tensor& Wh,  // [H,H]
                     const Tensor* B,   // [H] (optional)
                     const Tensor& h0,  // [N,H]
                     Tensor& Y,         // [N,T,H]
                     const RnnAttrs& attrs,
                     StreamHandle stream,
                     Tensor* Z_saved /*=nullptr*/,              // [N,T,H]
                     const RnnWorkspaceFwd* ws_fwd /*=nullptr*/);

// ========== Backward ==========
Status RnnCudaBackwardLaunch(const Tensor& X,       // [N,T,I]
                             const Tensor& Wx,      // [I,H]
                             const Tensor& Wh,      // [H,H]
                             const Tensor* B,       // [H] (optional)
                             const Tensor& h0,      // [N,H]
                             const Tensor& dY_post, // [N,T,H]
                             const Tensor& Z,       // [N,T,H]
                             Tensor* dWx,           // [I,H]
                             Tensor* dWh,           // [H,H]
                             Tensor* dB,            // [H]
                             Tensor* dh0,           // [N,H]
                             Tensor* dX,            // [N,T,I]
                             const RnnAttrs& attrs,
                             StreamHandle stream,
                             const RnnWorkspaceBwd* ws_bwd /*=nullptr*/);

} // namespace ai
