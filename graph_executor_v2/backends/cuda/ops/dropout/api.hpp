#pragma once
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct DropoutAttrs {
  float    p{0.1f};              // drop prob in [0,1)
  uint64_t seed{0x1234};         // stateless RNG seed
  bool     scale_in_train{true}; // true: 1/(1-p) scaling
  uint64_t counter_base{0};      // stateless RNG offset (for graph replay disambiguation)
};

Status DropoutCudaLaunch(const Tensor& X,
                         Tensor& Y,
                         Tensor* mask,            // may be null; if provided -> I32
                         const DropoutAttrs& attrs,
                         StreamHandle stream);

Status DropoutCudaBackwardLaunch(const Tensor& dY,
                                 const Tensor& mask, // I32
                                 Tensor& dX,
                                 const DropoutAttrs& attrs,
                                 StreamHandle stream);

} // namespace ai
