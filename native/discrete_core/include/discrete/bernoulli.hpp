#pragma once
#include <cstddef>
#include <cstdint>
#include "discrete_export.h"

namespace disc {
DISCRETE_API void bernoulli_cpu(float* out, std::size_t n,
                                float p, std::uint64_t seed=0);
DISCRETE_API void bernoulli_logpmf_cpu(const float* x, std::size_t n,
                                       float p, float* out);
#ifdef DISCRETE_WITH_CUDA
DISCRETE_API void bernoulli_cuda(float* out, std::size_t n,
                                 float p, std::uint64_t seed=0,
                                 void* stream=nullptr);
DISCRETE_API void bernoulli_logpmf_cuda(const float* x, std::size_t n,
                                        float p, float* out,
                                        void* stream=nullptr);
#endif
} // namespace disc
