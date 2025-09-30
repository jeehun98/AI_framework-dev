// Z 저장 ( pre activation ) 옵션 내용 구현 필요, 

#pragma once

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"


namespace ai {

// Forward (현재: f32, RowMajor, no-transpose, alpha=1, beta=0 전제)
ai::Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                          Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);

// Backward (현재: f32, RowMajor, no-transpose 전제)
ai::Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                            const Tensor& gY, const Tensor& Z,
                            Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                            const GemmAttrs& attrs, StreamHandle stream);

} // namespace ai
