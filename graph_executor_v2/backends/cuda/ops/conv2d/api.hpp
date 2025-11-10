#pragma once
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace ai {

struct Conv2DAttrs {
  int  stride_h{1}, stride_w{1};
  int  pad_h{0},    pad_w{0};
  int  dil_h{1},    dil_w{1};
  int  groups{1}; // MVP: 1만 지원
  ai::ActKind act{ai::ActKind::None};
  float       leaky_slope{0.01f};
  bool        with_bias{false};
  bool        save_z{false};
};

enum class ConvBiasKind : int { None=0, Scalar=1, PerC=2 };

// ========== 캡처-세이프 워크스페이스 ==========
struct Conv2DWorkspaceFwd {
  // 크기: K = Cin*Kh*Kw, HWo = H_out*W_out
  // [필수]
  float* dCol   {nullptr};   // [HWo, K]
  float* W_KC   {nullptr};   // [K, Cout]
  float* Y_tmp  {nullptr};   // [HWo, Cout]  (GEMM 결과 + epilogue D rows)
  // [옵션] save_z==true 일 때 필요
  float* Z_rows {nullptr};   // [HWo, Cout]  (pre-activation Z rows)
};

struct Conv2DWorkspaceBwd {
  // [필수]
  float* dCol    {nullptr};  // [HWo, K]
  float* dTmp    {nullptr};  // max(Cout*K, HWo*K)
  // [옵션] gX 경로 필요 시
  float* W_CK    {nullptr};  // [Cout, K]
  float* dY_HT   {nullptr};  // [HWo, Cout]
  // [옵션] gW 경로 필요 시
  float* dWpack  {nullptr};  // [Cout, K]
  // [필수] act backward 위해 필요
  float* gy_rows {nullptr};  // [Cout, HWo] (post-act grad 전치/보관 & dact 결과)
  float* Z_rows  {nullptr};  // [Cout, HWo] (pre-activation Z 전치)
};

// -------- Forward --------
Status Conv2DCudaLaunch(const Tensor& X,
                        const Tensor& W,
                        const Tensor* B,
                        Tensor& Y,
                        const Conv2DAttrs& attrs,
                        StreamHandle stream,
                        Tensor* Z_saved /*=nullptr*/,
                        const Conv2DWorkspaceFwd* ws_fwd /*=nullptr*/);

// -------- Backward --------
Status Conv2DCudaBackwardLaunch(const Tensor& X,
                                const Tensor& W,
                                const Tensor& dY,
                                const Tensor& Z,
                                Tensor* dW,
                                Tensor* dB,
                                Tensor* dX,
                                const Conv2DAttrs& attrs,
                                StreamHandle stream,
                                const Conv2DWorkspaceBwd* ws_bwd /*=nullptr*/);

} // namespace ai
