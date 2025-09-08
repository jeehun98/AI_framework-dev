#pragma once
#include <cstddef>
#include <cstdint>
#include "discrete_export.h"

namespace disc {
DISCRETE_API void binomial_cpu(float* out, std::size_t n_samples,
                               int n, float p, std::uint64_t seed=0);
DISCRETE_API void binomial_logpmf_cpu(const float* k, std::size_t n_k,
                                      int n, float p, float* out);
#ifdef DISCRETE_WITH_CUDA
DISCRETE_API void binomial_cuda(float* out, std::size_t n_samples,
                                int n, float p, std::uint64_t seed=0,
                                void* stream=nullptr);
DISCRETE_API void binomial_logpmf_cuda(const float* k, std::size_t n_k,
                                       int n, float p, float* out,
                                       void* stream=nullptr);
#endif
} // namespace disc
