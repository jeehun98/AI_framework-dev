#pragma once
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

// 필요 시 regemm_epilogue의 정책/에필로그 타입을 여기서 매핑

namespace ai::cuda_gemm {

template<ActKind A>
struct EpilogueFor;

template<>
struct EpilogueFor<ActKind::None> { /* using type = ...; */ };
template<>
struct EpilogueFor<ActKind::ReLU> { /* using type = ...; */ };
// GELU, Tanh 등 추가 매핑

} // namespace ai::cuda_gemm
