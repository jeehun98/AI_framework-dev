// backends/cuda/ops/gemm/api.hpp
#pragma once
#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"

namespace ai {
  ai::Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                            Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);
  ai::Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                              const Tensor& gY, const Tensor& Z,
                              Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                              const GemmAttrs& attrs, StreamHandle stream);
}
