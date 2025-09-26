#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

struct DropoutAttrs {
  float p{0.1f};          // drop prob in [0,1)
  uint64_t seed{0x1234};  // stateless RNG seed
  bool scale_in_train{true}; // true: 1/(1-p) scaling
  uint64_t counter_base{0};   // NEW: stateless RNG용 오프셋

};

// Forward: X:[M,N] f32 -> Y:[M,N] f32, mask:[M,N] i32 (optional)
Status DropoutCudaLaunch(const Tensor& X,
                         Tensor& Y,
                         Tensor* mask,            // may be null
                         const DropoutAttrs& attrs,
                         StreamHandle stream);

// Backward: dY:[M,N] f32, mask:[M,N] i32 -> dX:[M,N] f32
Status DropoutCudaBackwardLaunch(const Tensor& dY,
                                 const Tensor& mask,
                                 Tensor& dX,
                                 const DropoutAttrs& attrs,
                                 StreamHandle stream);

} // namespace ai
