// src/ops/rnn.cpp
#include "backends/cuda/ops/rnn/api.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp"

namespace ai {
namespace ops {

static int rnn_forward_dispatch(const Tensor& X, const Tensor& h0,
                                const Tensor& Wx, const Tensor& Wh, const Tensor* b,
                                Tensor& Hout, Tensor* Zbuf, const RNNAttrs& attrs,
                                StreamHandle s) {
  auto st = RNNCudaLaunch(X,h0,Wx,Wh,b,Hout,Zbuf,attrs,s);
  return st==Status::Ok ? 0 : -1;
}

static int rnn_backward_dispatch(const Tensor& X, const Tensor& Hout, const Tensor* Zbuf,
                                 const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                                 const Tensor& dHout,
                                 Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                                 const RNNAttrs& attrs, StreamHandle s) {
  auto st = RNNCudaBackwardLaunch(X,Hout,Zbuf,h0,Wx,Wh,dHout,dX,dh0,dWx,dWh,dB,attrs,s);
  return st==Status::Ok ? 0 : -1;
}

// (선택) 필요 시 Schema/Registry에 등록
struct RNNRegister {
  RNNRegister(){
    // 여기에 프레임워크 고유의 등록 로직을 연결하세요.
    // 예: Schema::register("rnn_forward", ...), Dispatcher::register_impl(...), 등
  }
} _rnn_register;

} // namespace ops
} // namespace ai
