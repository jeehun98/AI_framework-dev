#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

// X: 임의 stride, Y: row-major contiguous (dtype=F32)
// shape/stride는 Tensor.desc 기반
Status ContiguousCopyCudaLaunch(const Tensor& X, Tensor& Y, StreamHandle stream);



} // namespace ai
