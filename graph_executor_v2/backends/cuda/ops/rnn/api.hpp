// backends/cuda/ops/rnn/api.hpp
#pragma once
#include <cstdint>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct RNNAttrs {
  int T{0};
  int B{0};
  int I{0};
  int H{0};
  bool save_z{true}; // if true: Zbuf contains pre-activation Z
};

// ==== 여기 "호스트 함수 선언"이 반드시 있어야 launcher가 인지합니다 ====
Status fill_zero(Tensor& t, StreamHandle s);
Status add_bias_rowwise(Tensor& Y, const Tensor& b, int B, int H, StreamHandle s);
Status add_out(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s);
Status add_inplace(Tensor& A, const Tensor& B, StreamHandle s);
Status tanh_out(const Tensor& X, Tensor& Y, StreamHandle s);
Status tanh_bwd_from_out(const Tensor& Y, const Tensor& dY, Tensor& dZ, StreamHandle s);
Status rowwise_sum_accum(const Tensor& M, Tensor& out, int B, int H, StreamHandle s);
// ========================================================================


// Forward: X[M=TB, I], h0[B, H], Wx[I,H], Wh[H,H], (b[H] optional)
// Outputs: Hout[M, H], (Zbuf[M,H] optional if save_z)
Status RNNCudaLaunch(const Tensor& X, const Tensor& h0,
                     const Tensor& Wx, const Tensor& Wh,
                     const Tensor* b, Tensor& Hout, Tensor* Zbuf,
                     const RNNAttrs& attrs, StreamHandle s);

// Backward:
// Inputs: X[M,I], Hout[M,H], Zbuf[M,H] (nullable if !save_z), h0[B,H], Wx[I,H], Wh[H,H], dHout[M,H]
// Outputs: dX[M,I], dh0[B,H], dWx[I,H], dWh[H,H], dB[H]
Status RNNCudaBackwardLaunch(const Tensor& X, const Tensor& Hout, const Tensor* Zbuf,
                             const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                             const Tensor& dHout,
                             Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                             const RNNAttrs& attrs, StreamHandle s);

} // namespace ai
